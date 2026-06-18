import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from config.vqa_rad.stage2_eval_config_vqa_rad import Stage2EvalConfig
from datas.vqa_rad_datasets import VQARADDataset
from utils.qwen3vl.qwen3_vl_8B_quant_loader import Qwen3VLQuantizedLoader
from utils.data_tools.collator.vqa_rad.vqa_rad_datasets_eval_collator import VQARADEvalCollator
from utils.data_tools.prompt_cleaning.vqa_rad_answer_cleaning import vqa_rad_answer_eval_cleaning

from config.LLM_config import LLMAPIConfig
from LLM_api.gpt_5_mini import GPT5MiniClient
from LLM_api.prompts.vqa_rad_prompt_builder_gpt_5_mini import build_llm_judge_user_prompt, parse_llm_judge_response


def evaluate_vanilla_qwen(loader, processor, cfg, test_loader, llm_client, llm_cfg):
    """评测原生 Vanilla Qwen3-VL 的极简核心函数"""
    print(f"\n" + "=" * 50)
    print(f"🌟 开始评测 Vanilla Qwen3-VL (原生无微调基线)")
    print("=" * 50)

    # 1. 🚀 仅加载原生底座！彻底杜绝 LoRA 和 MoE 的污染
    model = loader.load_model()
    model.eval()
    print("✅ 原生底座加载完毕，无任何微调权重干扰！")

    # ================= Phase 1: 本地推理 =================
    metrics = {"total": 0, "norm_match": 0, "closed_total": 0, "closed_correct": 0, "open_total": 0, "open_correct": 0}
    all_records = []

    with torch.no_grad():
        for batch_inputs, metadata_list in tqdm(test_loader, desc="Local Inference"):
            # 将普通输入搬运至 GPU
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch_inputs.items()}

            # 💡 极其重要：安全剥离多模态专属字典 (原生模型不需要这些)
            inputs.pop("biomed_image_tensors", None)
            inputs.pop("biomed_text_tokens", None)

            # 🚀 原生 Generate 推理
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=cfg.do_sample,
                temperature=cfg.temperature
            )

            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in
                                     zip(inputs["input_ids"], generated_ids)]
            output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)

            for i, raw_pred in enumerate(output_texts):
                meta = metadata_list[i]
                gt_raw = meta["gt_answer"]
                ans_type = meta.get("answer_type", "").strip().upper()
                is_closed_question = (ans_type == "CLOSED")

                pred_norm = vqa_rad_answer_eval_cleaning(raw_pred)
                gt_norm = vqa_rad_answer_eval_cleaning(gt_raw)
                is_norm = (pred_norm == gt_norm)

                metrics["total"] += 1
                if is_norm: metrics["norm_match"] += 1
                if is_closed_question:
                    metrics["closed_total"] += 1
                    if is_norm: metrics["closed_correct"] += 1
                else:
                    metrics["open_total"] += 1
                    if is_norm: metrics["open_correct"] += 1

                all_records.append({
                    "question": meta["question"], "gt_raw": gt_raw, "gt_norm": gt_norm,
                    "pred_raw": raw_pred.strip(), "pred_norm": pred_norm,
                    "is_norm_match": is_norm, "question_category": "closed" if is_closed_question else "open"
                })

    # 卸载模型
    del model
    torch.cuda.empty_cache()

    # ================= Phase 2: LLM 裁判 =================
    semantic_rescued_strict = 0

    for record in tqdm(all_records, desc="LLM Judging"):
        if record["question_category"] == "open" and not record["is_norm_match"]:
            user_prompt = build_llm_judge_user_prompt(
                question=record["question"], gt_raw=record["gt_raw"], gt_norm=record["gt_norm"],
                pred_raw=record["pred_raw"], pred_norm=record["pred_norm"]
            )
            raw_response = llm_client.ask(llm_cfg.vqa_rad_llm_judge_system_prompt, user_prompt, temperature=0.0)
            parsed_result = parse_llm_judge_response(raw_response)

            if parsed_result["score"] == "correct":
                semantic_rescued_strict += 1

    open_semantic_strict_correct = metrics["open_correct"] + semantic_rescued_strict
    overall_strict_acc = (metrics["closed_correct"] + open_semantic_strict_correct) / metrics["total"] if metrics[
        "total"] else 0

    return {
        "checkpoint": "Vanilla_Qwen3-VL",
        "closed_acc": metrics["closed_correct"] / metrics["closed_total"] if metrics["closed_total"] else 0,
        "open_strict_acc": open_semantic_strict_correct / metrics["open_total"] if metrics["open_total"] else 0,
        "overall_strict_acc": overall_strict_acc
    }


def main():
    cfg = Stage2EvalConfig()

    # 🚀 强制关闭动态路由，防止加载没用的 BioMedCLIP 占用显存
    cfg.router_mode = "fixed"

    print("\n🔍 启动原生 Qwen3-VL 基线评测 (Zero-shot)...")

    # 1. 初始化底层环境
    loader = Qwen3VLQuantizedLoader(
        model_path=cfg.model_name_or_path, processor_path=cfg.model_name_or_path,
        load_in_4bit=cfg.load_in_4bit, bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant, bnb_4bit_compute_dtype=cfg.bnb_4bit_compute_dtype,
        torch_dtype=cfg.torch_dtype, attn_implementation=cfg.attn_implementation, device_map="auto"
    )
    processor = loader.load_processor()
    processor.tokenizer.padding_side = "left"

    # 2. 挂载原生 Eval Collator (不需要传 biomed_transform)
    test_dataset = VQARADDataset(jsonl_path=cfg.vqa_rad_test_jsonl_path, image_root=cfg.vqa_rad_image_root)
    eval_collator = VQARADEvalCollator(processor, cfg, None, None)
    test_loader = DataLoader(test_dataset, batch_size=cfg.per_device_eval_batch_size, collate_fn=eval_collator,
                             num_workers=cfg.dataloader_num_workers, shuffle=False)

    llm_cfg = LLMAPIConfig()
    llm_client = GPT5MiniClient(api_key=llm_cfg.gpt_5_mini_key, base_url=llm_cfg.base_url,
                                model=llm_cfg.judge_model_name)

    # 3. 评测并打印
    res = evaluate_vanilla_qwen(loader, processor, cfg, test_loader, llm_client, llm_cfg)

    print("\n\n" + "🏆" * 20 + " 原生基线结果 (Baseline) " + "🏆" * 20)
    print(f"| 模型 | Closed Acc (Yes/No) | Open Acc (Strict) | Overall Strict Acc |")
    print("| :--- | :---: | :---: | :---: |")
    print(
        f"| {res['checkpoint']} | {res['closed_acc']:.2%} | {res['open_strict_acc']:.2%} | {res['overall_strict_acc']:.2%} |")


if __name__ == "__main__":
    main()