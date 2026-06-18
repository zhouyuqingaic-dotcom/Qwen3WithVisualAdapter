import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

# 1. 🚀 导入 VQA-MED-2019 专属配置与数据集
from config.vqa_med_2019.stage2_eval_config_vqa_med_2019 import Stage2EvalConfig
from datas.vqa_med_2019_datasets import VQAMED2019Dataset

from utils.qwen3vl.qwen3_vl_8B_quant_loader import Qwen3VLQuantizedLoader

# 2. 🚀 导入 VQA-MED-2019 专属的 Eval Collator 和答案清洗器
from utils.data_tools.collator.vqa_med_2019.vqa_med_2019_eval_collator import VQAMED2019EvalCollator
from utils.data_tools.prompt_cleaning.vqa_med_2019_answer_cleaning import vqa_med_2019_answer_eval_cleaning

from config.LLM_config import LLMAPIConfig
from LLM_api.gpt_5_mini import GPT5MiniClient
from LLM_api.prompts.slake_prompt_builder_gpt_5_mini import build_llm_judge_user_prompt, parse_llm_judge_response


def evaluate_vanilla_qwen(loader, processor, cfg, test_loader, llm_client, llm_cfg):
    """评测原生 Vanilla Qwen3-VL 在 VQA-MED-2019 上的极简核心函数"""
    print(f"\n" + "=" * 50)
    print(f"🌟 开始评测 Vanilla Qwen3-VL (VQA-MED-2019 原生无微调基线)")
    print("=" * 50)

    # 1. 🚀 仅加载原生底座！彻底杜绝 LoRA 和 MoE 适配器的污染
    model = loader.load_model()
    model.eval()
    print("✅ 原生底座加载完毕，无任何微调权重干扰！")

    # ================= Phase 1: 本地推理 =================
    # 🎯 对齐全权重寻宝脚本，引入四大医疗类目细分追踪
    metrics = {
        "total": 0, "norm_match": 0,
        "closed_total": 0, "closed_correct": 0,
        "open_total": 0, "open_correct": 0,
        "Modality_total": 0, "Modality_correct": 0,
        "Plane_total": 0, "Plane_correct": 0,
        "Organ_total": 0, "Organ_correct": 0,
        "Abnormality_total": 0, "Abnormality_correct": 0,
    }
    all_records = []

    with torch.no_grad():
        for batch_inputs, metadata_list in tqdm(test_loader, desc="Local Inference"):
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch_inputs.items()}

            # 💡 极其重要：原生模型不需要多模态意图特征，安全剥离
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

                gt_raw = meta.get("gt_answer", "")
                ans_type = meta.get("answer_type", "").strip().upper()
                q_type = meta.get("question_type", "UNKNOWN")  # Modality, Plane, Organ, Abnormality
                is_closed_question = (ans_type == "CLOSED")

                pred_norm = vqa_med_2019_answer_eval_cleaning(raw_pred)
                gt_norm = vqa_med_2019_answer_eval_cleaning(gt_raw)
                is_norm = (pred_norm == gt_norm)

                metrics["total"] += 1
                if is_norm: metrics["norm_match"] += 1

                # 统计 Open / Closed
                if is_closed_question:
                    metrics["closed_total"] += 1
                    if is_norm: metrics["closed_correct"] += 1
                else:
                    metrics["open_total"] += 1
                    if is_norm: metrics["open_correct"] += 1

                # 🚀 统计四大类别精度
                if q_type in ["Modality", "Plane", "Organ", "Abnormality"]:
                    metrics[f"{q_type}_total"] += 1
                    if is_norm: metrics[f"{q_type}_correct"] += 1

                all_records.append({
                    "question": meta["question"], "gt_raw": gt_raw, "gt_norm": gt_norm,
                    "pred_raw": raw_pred.strip(), "pred_norm": pred_norm,
                    "is_norm_match": is_norm, "question_category": "closed" if is_closed_question else "open"
                })

    # 彻底释放原生底座，拒绝显存交叉污染
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
        "overall_strict_acc": overall_strict_acc,
        "modality_acc": metrics["Modality_correct"] / metrics["Modality_total"] if metrics["Modality_total"] else 0,
        "plane_acc": metrics["Plane_correct"] / metrics["Plane_total"] if metrics["Plane_total"] else 0,
        "organ_acc": metrics["Organ_correct"] / metrics["Organ_total"] if metrics["Organ_total"] else 0,
        "abnormality_acc": metrics["Abnormality_correct"] / metrics["Abnormality_total"] if metrics[
            "Abnormality_total"] else 0,
    }


def main():
    cfg = Stage2EvalConfig()

    # 🚀 强制关闭动态路由，防止加载没用的 BioMedCLIP 占用零刷基线显存
    cfg.router_mode = "fixed"

    print("\n🔍 启动 VQA-MED-2019 原生 Qwen3-VL 基线评测 (Zero-shot)...")

    # 1. 初始化底层环境
    loader = Qwen3VLQuantizedLoader(
        model_path=cfg.model_name_or_path, processor_path=cfg.model_name_or_path,
        load_in_4bit=cfg.load_in_4bit, bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant, bnb_4bit_compute_dtype=cfg.bnb_4bit_compute_dtype,
        torch_dtype=cfg.torch_dtype, attn_implementation=cfg.attn_implementation, device_map="auto"
    )
    processor = loader.load_processor()
    processor.tokenizer.padding_side = "left"

    # 2. 挂载原生 Eval Collator (不传任何多模态意图提取引擎)
    test_dataset = VQAMED2019Dataset(data_path=cfg.vqa_med_2019_test_data_path,
                                     image_root=cfg.vqa_med_2019_test_image_root)
    eval_collator = VQAMED2019EvalCollator(processor, cfg, None, None)
    test_loader = DataLoader(test_dataset, batch_size=cfg.per_device_eval_batch_size, collate_fn=eval_collator,
                             num_workers=cfg.dataloader_num_workers, shuffle=False)

    llm_cfg = LLMAPIConfig()
    llm_client = GPT5MiniClient(api_key=llm_cfg.gpt_5_mini_key, base_url=llm_cfg.base_url,
                                model=llm_cfg.judge_model_name)

    # 3. 评测并激活 8 列豪华排版
    res = evaluate_vanilla_qwen(loader, processor, cfg, test_loader, llm_client, llm_cfg)

    print("\n\n" + "🏆" * 20 + " VQA-MED-2019 原生基线结果 (Baseline) " + "🏆" * 20)
    print(f"| 模型 | Modality | Plane | Organ | Abnormality | Closed | Open (Strict) | Overall |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    print(
        f"| {res['checkpoint']} | {res['modality_acc']:.2%} | {res['plane_acc']:.2%} | {res['organ_acc']:.2%} | {res['abnormality_acc']:.2%} | {res['closed_acc']:.2%} | {res['open_strict_acc']:.2%} | **{res['overall_strict_acc']:.2%}** |")


if __name__ == "__main__":
    main()