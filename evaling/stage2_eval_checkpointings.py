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
from utils.data_tools.prompt_cleaning.vqa_rad_answer_cleaning import vqa_rad_answer_eval_cleaning

# 🚀 导入 MoE 的终极武器库
from utils.qwen3vl.qwen3_vl_8B_visual_adapter import VisualAdapter_Global, VisualAdapter_Local, VisualAdapter_Region
from utils.qwen3vl.qwen3_vl_8B_visual_adapters_fusion import Qwen3VLMoEVisualAdapterDynamicFusion, \
    Qwen3VLMoEVisualAdapterFixedFusion
from utils.biomedclip.biomed_clip_loader import load_biomedclip

from config.LLM_config import LLMAPIConfig
from LLM_api.gpt_5_mini import GPT5MiniClient
from LLM_api.prompts.vqa_rad_prompt_builder_gpt_5_mini import build_llm_judge_user_prompt, parse_llm_judge_response


def evaluate_single_checkpoint(weights_path, loader, processor, cfg, test_loader, llm_client, llm_cfg,
                               biomed_extractor=None):
    """评测单个 Checkpoint 的核心函数"""
    print(f"\n" + "=" * 50)
    print(f"🌟 开始评测 Checkpoint: {os.path.basename(weights_path)}")
    print("=" * 50)

    # 1. 每次都获取一个极其干净的底座！彻底杜绝 LoRA 堆叠污染和 Warning
    base_model = loader.load_model()

    # 直接用 PeftModel 从对应的 checkpoint 读取微调后的 LoRA 权重
    model = PeftModel.from_pretrained(base_model, weights_path)

    # =========================================================
    # 2. 🚀 手工挂载 MoE 视觉残差架构 (模型夺舍)
    # =========================================================
    vision_tower = model.base_model.model.model.visual
    ref_param = next(vision_tower.parameters())
    adapter_dtype = ref_param.dtype if ref_param.is_floating_point() else torch.bfloat16

    # 实例化三大专家
    adapter_global = VisualAdapter_Global(hidden_dim=cfg.visual_adapter_hidden_dim, r=cfg.visual_adapter_r,
                                          kernel_size=cfg.global_adapter_kernel_size)
    adapter_local = VisualAdapter_Local(hidden_dim=cfg.visual_adapter_hidden_dim, r=cfg.visual_adapter_r,
                                        kernel_size=cfg.local_adapter_kernel_size)
    adapter_region = VisualAdapter_Region(hidden_dim=cfg.visual_adapter_hidden_dim, r=cfg.visual_adapter_r,
                                          kernel_size=cfg.region_adapter_kernel_size)

    # 实例化 Fusion 层并依赖注入
    if cfg.router_mode == "dynamic":
        fusion_layer = Qwen3VLMoEVisualAdapterDynamicFusion(
            hidden_dim=cfg.visual_adapter_hidden_dim,
            adapter_global=adapter_global,
            adapter_local=adapter_local,
            adapter_region=adapter_region
        )
        model.biomed_extractor = biomed_extractor.to(device=ref_param.device, dtype=adapter_dtype)
    else:
        fusion_layer = Qwen3VLMoEVisualAdapterFixedFusion(
            hidden_dim=cfg.visual_adapter_hidden_dim,
            adapter_global=adapter_global,
            adapter_local=adapter_local,
            adapter_region=adapter_region,
            fixed_weights=cfg.fixed_weights
        )

    # 上户口
    vision_tower.res_adapter = fusion_layer.to(device=ref_param.device, dtype=adapter_dtype)

    # 载入 Stage 2 专属的 MoE 权重
    adapter_pt_path = os.path.join(weights_path, "visual_adapter.pt")
    if not os.path.exists(adapter_pt_path):
        print(f"⚠️ 跳过: 找不到 Adapter 权重 {adapter_pt_path}")
        del model
        del base_model
        torch.cuda.empty_cache()
        return None

    vision_tower.res_adapter.load_state_dict(torch.load(adapter_pt_path, map_location=ref_param.device))
    print(f"✅ 成功接驳 Stage 2 多尺度 MoE 视觉适配器！融合模式: {cfg.router_mode.upper()}")

    # =========================================================
    # 3. 🚀 内层劫持 Forward (适配 Generate 的极限环境)
    # =========================================================
    vision_tower.original_forward = vision_tower.forward

    def patched_vision_forward(self, *args, **kwargs):
        outputs = self.original_forward(*args, **kwargs)

        # 💡 这里非常关键：直接从 vision_tower 自身获取特征
        img_f = getattr(self, "current_biomed_img_feat", None)
        txt_f = getattr(self, "current_biomed_txt_feat", None)

        grid_thw = kwargs.get("grid_thw", None)
        if grid_thw is None and len(args) > 1:
            grid_thw = args[1]

        is_dynamic = (img_f is not None and txt_f is not None)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            if is_dynamic:
                outputs.pooler_output = self.res_adapter(outputs.pooler_output, biomed_img_feat=img_f,
                                                         biomed_txt_feat=txt_f, grid_thw=grid_thw)
            else:
                outputs.pooler_output = self.res_adapter(outputs.pooler_output, grid_thw=grid_thw)

        if hasattr(outputs, "deepstack_features") and outputs.deepstack_features is not None:
            if is_dynamic:
                outputs.deepstack_features = [
                    self.res_adapter(x, biomed_img_feat=img_f, biomed_txt_feat=txt_f, grid_thw=grid_thw) for x in
                    outputs.deepstack_features]
            else:
                outputs.deepstack_features = [self.res_adapter(x, grid_thw=grid_thw) for x in
                                              outputs.deepstack_features]
        return outputs

    vision_tower.forward = types.MethodType(patched_vision_forward, vision_tower)

    model.eval()

    # ================= Phase 1: 本地推理 =================
    metrics = {"total": 0, "norm_match": 0, "closed_total": 0, "closed_correct": 0, "open_total": 0, "open_correct": 0}
    all_records = []

    with torch.no_grad():
        for batch_inputs, metadata_list in tqdm(test_loader, desc="Local Inference"):
            # 将普通输入搬运至 GPU
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch_inputs.items()}

            # 💡 极其重要：安全剥离多模态专属字典，避开 HF 的类型审查
            biomed_img = inputs.pop("biomed_image_tensors", None)
            biomed_txt = inputs.pop("biomed_text_tokens", None)

            # 🚀 绕过 generate() 的偷天换日：手动算好特征并直接挂在 vision_tower 上
            if cfg.router_mode == "dynamic" and biomed_img is not None and biomed_txt is not None:
                biomed_img = biomed_img.to(dtype=adapter_dtype)
                img_f, txt_f = model.biomed_extractor(biomed_img, biomed_txt)
                vision_tower.current_biomed_img_feat = img_f
                vision_tower.current_biomed_txt_feat = txt_f
            else:
                vision_tower.current_biomed_img_feat = None
                vision_tower.current_biomed_txt_feat = None

            # generate 不会再抱怨类型错误了！
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

    # 1. 动态获取权重根目录 (完全听从 Config 自动生成的 stage2_weights_dir)
    base_weight_dir = os.path.dirname(cfg.stage2_weights_dir)
    checkpoint_dirs = glob.glob(os.path.join(base_weight_dir, "checkpoint-*"))
    checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]))

    final_weights_path = os.path.join(base_weight_dir, "final_weights")
    if os.path.exists(final_weights_path):
        checkpoint_dirs.append(final_weights_path)

    if not checkpoint_dirs:
        print(f"❌ 在 {base_weight_dir} 下没有找到任何 checkpoint 文件夹！")
        return

    print(f"🔍 寻宝启动！评测模式: [MoE {cfg.router_mode.upper()}] | 共发现 {len(checkpoint_dirs)} 个节点：")
    for cp in checkpoint_dirs: print(f"  - {os.path.basename(cp)}")

    # 2. 初始化底层环境
    loader = Qwen3VLQuantizedLoader(
        model_path=cfg.model_name_or_path, processor_path=cfg.model_name_or_path,
        load_in_4bit=cfg.load_in_4bit, bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant, bnb_4bit_compute_dtype=cfg.bnb_4bit_compute_dtype,
        torch_dtype=cfg.torch_dtype, attn_implementation=cfg.attn_implementation, device_map="auto"
    )
    processor = loader.load_processor()
    processor.tokenizer.padding_side = "left"

    # 3. 🧠 初始化 BioMedCLIP 大脑 (提前加载一次，避免循环时重复加载造成开销爆表)
    biomed_extractor, biomed_transform, biomed_tokenizer = None, None, None
    if cfg.router_mode == "dynamic":
        biomed_extractor, biomed_transform, biomed_tokenizer = load_biomedclip(cfg.biomedclip_path)

    # 4. 挂载带意图截获的专属 Eval Collator
    test_dataset = VQARADDataset(jsonl_path=cfg.vqa_rad_test_jsonl_path, image_root=cfg.vqa_rad_image_root)
    eval_collator = VQARADEvalCollator(processor, cfg, biomed_transform, biomed_tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=cfg.per_device_eval_batch_size, collate_fn=eval_collator,
                             num_workers=cfg.dataloader_num_workers, shuffle=False)

    llm_cfg = LLMAPIConfig()
    llm_client = GPT5MiniClient(api_key=llm_cfg.gpt_5_mini_key, base_url=llm_cfg.base_url,
                                model=llm_cfg.judge_model_name)

    # 5. 循环评测
    results = []
    for cp_path in checkpoint_dirs:
        res = evaluate_single_checkpoint(
            cp_path, loader, processor, cfg, test_loader, llm_client, llm_cfg, biomed_extractor=biomed_extractor
        )
        if res: results.append(res)

    # 6. 打印 Markdown 排行榜
    print("\n\n" + "🏆" * 20 + " 寻宝结果 (Leaderboard) " + "🏆" * 20)
    print(
        f"| 评测节点 (模式: {cfg.router_mode.upper()}) | Closed Acc (Yes/No) | Open Acc (Strict) | Overall Strict Acc |")
    print("| :--- | :---: | :---: | :---: |")
    for res in results:
        print(
            f"| {res['checkpoint']} | {res['closed_acc']:.2%} | {res['open_strict_acc']:.2%} | {res['overall_strict_acc']:.2%} |")

    best_overall = max(results, key=lambda x: x["overall_strict_acc"])
    print(f"\n🎯 恭喜！找到黄金节点：**{best_overall['checkpoint']}** (Overall: {best_overall['overall_strict_acc']:.2%})")


if __name__ == "__main__":
    main()