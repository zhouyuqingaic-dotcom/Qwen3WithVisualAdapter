import os
import glob
import types
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from peft import PeftModel

from config.stage2_eval_config import Stage2EvalConfig
from datas.vqa_rad_datasets import VQARADDataset
from utils.qwen3vl.qwen3_vl_8B_quant_loader import Qwen3VLQuantizedLoader
from utils.data_tools.collator.vqa_rad_datasets_eval_collator import VQARADEvalCollator
from utils.qwen3vl.qwen3_vl_8B_visual_adapter import Qwen3VLVisualAdapter
from utils.data_tools.prompt_cleaning.vqa_rad_answer_cleaning import vqa_rad_answer_eval_cleaning

from config.LLM_config import LLMAPIConfig
from LLM_api.gpt_5_mini import GPT5MiniClient
from LLM_api.prompts.vqa_rad_prompt_builder_gpt_5_mini import build_llm_judge_user_prompt, parse_llm_judge_response


def evaluate_single_checkpoint(weights_path, loader, processor, cfg, test_loader, llm_client, llm_cfg):
    """评测单个 Checkpoint 的核心函数"""
    print(f"\n" + "=" * 50)
    print(f"🌟 开始评测 Checkpoint: {os.path.basename(weights_path)}")
    print("=" * 50)

    # 1. 每次都获取一个极其干净的底座！彻底杜绝 LoRA 堆叠污染和 Warning
    base_model = loader.load_model()
    model = PeftModel.from_pretrained(base_model, weights_path)

    # 2. 如果开启了视觉适配器，则挂载它
    if cfg.use_visual_adapter:
        vision_tower = model.base_model.model.model.visual
        ref_param = next(vision_tower.parameters())
        adapter_dtype = ref_param.dtype if ref_param.is_floating_point() else torch.bfloat16

        adapter = Qwen3VLVisualAdapter(
            hidden_dim=cfg.visual_adapter_hidden_dim,
            r=cfg.visual_adapter_r,
            init_alpha=cfg.visual_adapter_alpha,
        ).to(device=ref_param.device, dtype=adapter_dtype)

        adapter_pt_path = os.path.join(weights_path, "visual_adapter.pt")
        if not os.path.exists(adapter_pt_path):
            print(f"⚠️ 跳过: 找不到 Adapter 权重 {adapter_pt_path}")
            del model
            del base_model
            torch.cuda.empty_cache()
            return None

        adapter.load_state_dict(torch.load(adapter_pt_path, map_location=ref_param.device))
        vision_tower.res_adapter = adapter

        def patched_forward(self, *args, **kwargs):
            outputs = self.original_forward(*args, **kwargs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                outputs.pooler_output = self.res_adapter(outputs.pooler_output)
            if hasattr(outputs, "deepstack_features") and outputs.deepstack_features is not None:
                outputs.deepstack_features = [self.res_adapter(x) for x in outputs.deepstack_features]
            return outputs

        vision_tower.original_forward = vision_tower.forward
        vision_tower.forward = types.MethodType(patched_forward, vision_tower)
    else:
        print("💡 当前为纯 LoRA Baseline 模式，跳过 Visual Adapter 挂载。")

    model.eval()

    # ================= Phase 1: 本地推理 =================
    metrics = {"total": 0, "norm_match": 0, "closed_total": 0, "closed_correct": 0, "open_total": 0, "open_correct": 0}
    all_records = []

    with torch.no_grad():
        for batch_inputs, metadata_list in tqdm(test_loader, desc="Local Inference"):
            inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}
            generated_ids = model.generate(**inputs, max_new_tokens=cfg.max_new_tokens, do_sample=cfg.do_sample,
                                           temperature=cfg.temperature)
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

    # 卸载本轮模型释放显存
    del model
    del base_model
    torch.cuda.empty_cache()

    # ================= Phase 2: LLM 裁判 =================
    semantic_rescued_strict = 0
    semantic_rescued_relaxed = 0

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
                semantic_rescued_relaxed += 1
            elif parsed_result["score"] == "partially_correct":
                semantic_rescued_relaxed += 1

    open_semantic_strict_correct = metrics["open_correct"] + semantic_rescued_strict
    overall_strict_acc = (metrics["closed_correct"] + open_semantic_strict_correct) / metrics["total"] if metrics[
        "total"] else 0

    return {
        "checkpoint": os.path.basename(weights_path),
        "closed_acc": metrics["closed_correct"] / metrics["closed_total"] if metrics["closed_total"] else 0,
        "open_strict_acc": open_semantic_strict_correct / metrics["open_total"] if metrics["open_total"] else 0,
        "overall_strict_acc": overall_strict_acc
    }


def main():
    cfg = Stage2EvalConfig()

    # 1. 动态获取权重根目录 (完全听从 Config 指挥)
    if cfg.use_visual_adapter:
        base_weight_dir = os.path.dirname(cfg.stage2_weights_with_adapter)
    else:
        base_weight_dir = os.path.dirname(cfg.stage2_weights_baseline)

    checkpoint_dirs = glob.glob(os.path.join(base_weight_dir, "checkpoint-*"))
    checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]))

    final_weights_path = os.path.join(base_weight_dir, "final_weights")
    if os.path.exists(final_weights_path):
        checkpoint_dirs.append(final_weights_path)

    if not checkpoint_dirs:
        print(f"❌ 在 {base_weight_dir} 下没有找到任何 checkpoint 文件夹！")
        return

    print(
        f"🔍 寻宝启动！评测模式: {'[开启 Adapter]' if cfg.use_visual_adapter else '[纯 LoRA]'} | 共发现 {len(checkpoint_dirs)} 个节点：")
    for cp in checkpoint_dirs: print(f"  - {os.path.basename(cp)}")

    # 2. 初始化环境
    loader = Qwen3VLQuantizedLoader(
        model_path=cfg.model_name_or_path, processor_path=cfg.model_name_or_path,
        load_in_4bit=cfg.load_in_4bit, bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant, bnb_4bit_compute_dtype=cfg.bnb_4bit_compute_dtype,
        torch_dtype=cfg.torch_dtype, attn_implementation=cfg.attn_implementation, device_map="auto"
    )
    processor = loader.load_processor()
    processor.tokenizer.padding_side = "left"

    test_dataset = VQARADDataset(jsonl_path=cfg.vqa_rad_test_jsonl_path, image_root=cfg.vqa_rad_image_root)
    eval_collator = VQARADEvalCollator(processor, cfg)
    test_loader = DataLoader(test_dataset, batch_size=cfg.per_device_eval_batch_size, collate_fn=eval_collator,
                             num_workers=cfg.dataloader_num_workers, shuffle=False)

    llm_cfg = LLMAPIConfig()
    llm_client = GPT5MiniClient(api_key=llm_cfg.gpt_5_mini_key, base_url=llm_cfg.base_url,
                                model=llm_cfg.judge_model_name)

    # 3. 循环评测
    results = []
    for cp_path in checkpoint_dirs:
        # 注意这里把 loader 传进去了，每次都在里面重新取用新模型
        res = evaluate_single_checkpoint(cp_path, loader, processor, cfg, test_loader, llm_client, llm_cfg)
        if res: results.append(res)

    # 4. 打印 Markdown 排行榜
    print("\n\n" + "🏆" * 20 + " 寻宝结果 (Leaderboard) " + "🏆" * 20)
    print("| 评测节点 (Checkpoint) | Closed Acc (Yes/No) | Open Acc (Strict) | Overall Strict Acc |")
    print("| :--- | :---: | :---: | :---: |")
    for res in results:
        print(
            f"| {res['checkpoint']} | {res['closed_acc']:.2%} | {res['open_strict_acc']:.2%} | {res['overall_strict_acc']:.2%} |")

    best_overall = max(results, key=lambda x: x["overall_strict_acc"])
    print(f"\n🎯 恭喜！找到黄金节点：**{best_overall['checkpoint']}** (Overall: {best_overall['overall_strict_acc']:.2%})")


if __name__ == "__main__":
    main()