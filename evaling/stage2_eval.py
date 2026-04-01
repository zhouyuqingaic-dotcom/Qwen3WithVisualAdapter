import os
import sys
import json
import types
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from peft import PeftModel

# 确保脚本能够找到根目录下的自定义包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.stage2_eval_config import Stage2EvalConfig
from datas.vqa_rad_datasets import VQARADDataset
from utils.qwen3vl.qwen3_vl_8B_quant_loader import Qwen3VLQuantizedLoader
from utils.data_tools.collator.vqa_rad_datasets_eval_collator import VQARADEvalCollator
from utils.qwen3vl.qwen3_vl_8B_visual_adapter import Qwen3VLVisualAdapter
from utils.data_tools.prompt_cleaning.vqa_rad_answer_cleaning import vqa_rad_answer_eval_cleaning

# 导入 LLM 裁判相关模块
from config.LLM_config import LLMAPIConfig
from LLM_api.gpt_5_mini import GPT5MiniClient
from LLM_api.prompts.vqa_rad_prompt_builder_gpt_5_mini import build_llm_judge_user_prompt, parse_llm_judge_response


def main():
    cfg = Stage2EvalConfig()

    print(f"\n🚀 [1/6] 启动 Stage 2 终极评估！结果将保存至: {cfg.output_dir}")

    # 动态获取权重路径
    weights_path = cfg.stage2_weights_with_adapter if cfg.use_visual_adapter else cfg.stage2_weights_baseline
    print(f"🎯 当前评测模式: {'[增强版] 带 Visual Adapter' if cfg.use_visual_adapter else '[对照组] 纯 LoRA Baseline'}")
    print(f"📂 读取权重路径: {weights_path}")

    # =========================================================================
    # 1. 加载底座模型与 Processor
    # =========================================================================
    print("\n⏳ [2/6] 正在加载 Qwen3-VL 4-bit 底座模型...")
    loader = Qwen3VLQuantizedLoader(
        model_path=cfg.model_name_or_path,
        processor_path=cfg.model_name_or_path,
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=cfg.bnb_4bit_compute_dtype,
        torch_dtype=cfg.torch_dtype,
        attn_implementation=cfg.attn_implementation,
        device_map="auto",
    )
    base_model, processor = loader.load()
    processor.tokenizer.padding_side = "left"

    # =========================================================================
    # 2. 挂载 Stage 2 权重 (LoRA + Visual Adapter)
    # =========================================================================
    print(f"\n⏳ [3/6] 正在装载目标权重...")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"找不到权重: {weights_path}")

    # 2.1 挂载 LoRA
    model = PeftModel.from_pretrained(base_model, weights_path)

    # 2.2 挂载 Visual Adapter (如果启用)
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
        adapter.load_state_dict(torch.load(adapter_pt_path, map_location=ref_param.device))

        vision_tower.res_adapter = adapter
        vision_tower.original_forward = vision_tower.forward

        def patched_forward(self, *args, **kwargs):
            outputs = self.original_forward(*args, **kwargs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                outputs.pooler_output = self.res_adapter(outputs.pooler_output)
            if hasattr(outputs, "deepstack_features") and outputs.deepstack_features is not None:
                outputs.deepstack_features = [self.res_adapter(x) for x in outputs.deepstack_features]
            return outputs

        vision_tower.forward = types.MethodType(patched_forward, vision_tower)
        print("✨ 视觉残差适配器 (Visual Adapter) 成功加载并接驳！")

    model.eval()

    # =========================================================================
    # 3. 准备数据与 DataLoader
    # =========================================================================
    print(f"\n⏳ [4/6] 准备 VQA-RAD 官方测试集: {cfg.vqa_rad_test_jsonl_path}")
    test_dataset = VQARADDataset(jsonl_path=cfg.vqa_rad_test_jsonl_path, image_root=cfg.vqa_rad_image_root)
    eval_collator = VQARADEvalCollator(processor, cfg)

    test_loader = DataLoader(
        test_dataset, batch_size=cfg.per_device_eval_batch_size,
        collate_fn=eval_collator, num_workers=cfg.dataloader_num_workers, shuffle=False
    )

    # =========================================================================
    # 4. Phase 1: 本地推理与官方硬规则打分
    # =========================================================================
    print("\n🔥 [5/6] Phase 1: 开始模型推理与官方标准(CLOSED/OPEN)打分...")
    metrics = {
        "total": 0, "exact_match": 0, "norm_match": 0,
        "closed_total": 0, "closed_correct": 0,
        "open_total": 0, "open_correct": 0
    }
    all_records = []

    with torch.no_grad():
        for batch_inputs, metadata_list in tqdm(test_loader, desc="Local Inference"):
            inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}

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

                # 🎓 学术级评测：直接读取官方原始数据提供的 answer_type
                ans_type = meta.get("answer_type", "").strip().upper()
                is_closed_question = (ans_type == "CLOSED")

                is_exact = (raw_pred.strip() == gt_raw.strip())
                pred_norm = vqa_rad_answer_eval_cleaning(raw_pred)
                gt_norm = vqa_rad_answer_eval_cleaning(gt_raw)
                is_norm = (pred_norm == gt_norm)

                metrics["total"] += 1
                if is_exact: metrics["exact_match"] += 1
                if is_norm: metrics["norm_match"] += 1

                # 严格按照官方 Closed / Open 进行计分
                if is_closed_question:
                    metrics["closed_total"] += 1
                    if is_norm: metrics["closed_correct"] += 1
                else:
                    metrics["open_total"] += 1
                    if is_norm: metrics["open_correct"] += 1

                all_records.append({
                    "index": meta["index"],
                    "image_path": meta["image_path"],
                    "question": meta["question"],
                    "gt_raw": gt_raw,
                    "gt_norm": gt_norm,
                    "pred_raw": raw_pred.strip(),
                    "pred_norm": pred_norm,
                    "is_exact_match": is_exact,
                    "is_norm_match": is_norm,
                    "question_category": "closed" if is_closed_question else "open",
                    "llm_score": None, "llm_reasoning": None
                })

    del model
    torch.cuda.empty_cache()

    # =========================================================================
    # 5. Phase 2: LLM 裁判抢救流程 (Semantic Rescue)
    # =========================================================================
    print("\n🤖 [6/6] Phase 2: 呼叫云端 LLM 裁判，对错判的 Open 题进行语义抢救...")
    llm_cfg = LLMAPIConfig()
    llm_client = GPT5MiniClient(api_key=llm_cfg.gpt_5_mini_key, base_url=llm_cfg.base_url,
                                model=llm_cfg.judge_model_name)

    semantic_rescued_strict = 0
    semantic_rescued_relaxed = 0

    for record in tqdm(all_records, desc="LLM Judging"):
        # 仅对 open 题且硬规则算错的进行云端抢救
        if record["question_category"] == "open" and not record["is_norm_match"]:
            user_prompt = build_llm_judge_user_prompt(
                question=record["question"],
                gt_raw=record["gt_raw"],
                gt_norm=record["gt_norm"],
                pred_raw=record["pred_raw"],
                pred_norm=record["pred_norm"]
            )
            raw_response = llm_client.ask(llm_cfg.vqa_rad_llm_judge_system_prompt, user_prompt, temperature=0.0)
            parsed_result = parse_llm_judge_response(raw_response)

            record["llm_score"] = parsed_result["score"]
            record["llm_reasoning"] = parsed_result["reasoning"]

            if parsed_result["score"] == "correct":
                semantic_rescued_strict += 1
                semantic_rescued_relaxed += 1
            elif parsed_result["score"] == "partially_correct":
                semantic_rescued_relaxed += 1
        else:
            if record["is_norm_match"]:
                record["llm_score"] = "correct"
                record["llm_reasoning"] = "Rule-based Match (Bypassed LLM)"
            else:
                record["llm_score"] = "incorrect"
                record["llm_reasoning"] = "Rule-based Mismatch on Closed Question (Bypassed LLM)"

    # =========================================================================
    # 6. 保存明细与汇总报告
    # =========================================================================
    details_file_path = os.path.join(cfg.output_dir, "eval_details_vqa_rad_official.jsonl")
    with open(details_file_path, "w", encoding="utf-8") as f_out:
        for rec in all_records:
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    open_semantic_strict_correct = metrics["open_correct"] + semantic_rescued_strict
    open_semantic_relaxed_correct = metrics["open_correct"] + semantic_rescued_relaxed

    # 计算 Overall 分数
    overall_hard_acc = metrics["norm_match"] / metrics["total"] if metrics["total"] else 0
    overall_strict_acc = (metrics["closed_correct"] + open_semantic_strict_correct) / metrics["total"] if metrics[
        "total"] else 0
    overall_relaxed_acc = (metrics["closed_correct"] + open_semantic_relaxed_correct) / metrics["total"] if metrics[
        "total"] else 0

    summary = {
        "model_mode": "Visual_Adapter_Enhanced" if cfg.use_visual_adapter else "LoRA_Only_Baseline",
        "total_samples": metrics["total"],
        "overall_accuracy": {
            "hard_match": overall_hard_acc,
            "semantic_strict": overall_strict_acc,
            "semantic_relaxed": overall_relaxed_acc,
        },
        "closed_questions": {
            "total": metrics["closed_total"],
            "accuracy": metrics["closed_correct"] / metrics["closed_total"] if metrics["closed_total"] else 0
        },
        "open_questions": {
            "total": metrics["open_total"],
            "rule_based_accuracy": metrics["open_correct"] / metrics["open_total"] if metrics["open_total"] else 0,
            "semantic_strict_accuracy": open_semantic_strict_correct / metrics["open_total"] if metrics[
                "open_total"] else 0,
            "semantic_relaxed_accuracy": open_semantic_relaxed_correct / metrics["open_total"] if metrics[
                "open_total"] else 0,
            "semantic_rescue_count": semantic_rescued_strict
        }
    }

    summary_path = os.path.join(cfg.output_dir, "eval_summary_vqa_rad_official.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"🎉 官方口径评测完成！当前模式: [{summary['model_mode']}]")
    print("-" * 60)
    print(f"🟢 Closed 题准度:       {summary['closed_questions']['accuracy']:.2%} (共 {metrics['closed_total']} 题)")
    print(
        f"🔵 Open 题 (硬规则准度): {summary['open_questions']['rule_based_accuracy']:.2%} (共 {metrics['open_total']} 题)")
    print(
        f"🌟 Open 题 (Strict):   {summary['open_questions']['semantic_strict_accuracy']:.2%} 🚀 (LLM 抢救 {semantic_rescued_strict} 题)")
    print(f"🌟 Open 题 (Relaxed):  {summary['open_questions']['semantic_relaxed_accuracy']:.2%}")
    print("-" * 60)
    print(f"🏆 Overall (Hard):     {overall_hard_acc:.2%}")
    print(f"🏆 Overall (Strict):   {overall_strict_acc:.2%}")
    print(f"🏆 Overall (Relaxed):  {overall_relaxed_acc:.2%}")
    print("=" * 60)
    print(f"📂 裁判明细: {details_file_path}")
    print(f"📊 汇总报告: {summary_path}")


if __name__ == "__main__":
    main()