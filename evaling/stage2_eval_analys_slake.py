import os
import glob
import types
import torch
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader
from peft import PeftModel

# 1. 🚀 导入 SLAKE 专属配置与数据集
from config.slake.stage2_eval_config_slake import Stage2EvalConfig
from datas.slake_datasets import SLAKEDataset
from utils.qwen3vl.qwen3_vl_8B_quant_loader import Qwen3VLQuantizedLoader

# 2. 🚀 导入 SLAKE 专属的 Eval Collator 和答案清洗器
from utils.data_tools.collator.slake.slake_datasets_eval_collator import SLAKEEvalCollator
from utils.data_tools.prompt_cleaning.slake_answer_cleaning import slake_answer_eval_cleaning

# 3. 🚀 导入 MoE 的核心武器库
from utils.qwen3vl.qwen3_vl_8B_visual_adapter import VisualAdapter_Global, VisualAdapter_Local, VisualAdapter_Region
from utils.qwen3vl.qwen3_vl_8B_visual_adapters_fusion import Qwen3VLMoEVisualAdapterDynamicFusion, \
    Qwen3VLMoEVisualAdapterFixedFusion
from utils.biomedclip.biomed_clip_loader import load_biomedclip

# 4. 🚀 导入 LLM 裁判组件
from config.LLM_config import LLMAPIConfig
from LLM_api.gpt_5_mini import GPT5MiniClient
from LLM_api.prompts.slake_prompt_builder_gpt_5_mini import build_llm_judge_user_prompt, parse_llm_judge_response


def process_single_checkpoint_to_log(weights_path, loader, processor, cfg, test_loader, llm_client, llm_cfg,
                                     biomed_extractor=None):
    """评测单个 Checkpoint，完全复刻 checkpoints 脚本的 Phase 1 和 Phase 2，并输出带 w 值的打分日志"""
    cp_name = os.path.basename(weights_path)
    print(f"\n" + "=" * 60)
    print(f"🌟 正在多节点深挖寻宝 Checkpoint: {cp_name}")
    print("=" * 60)

    # 📝 动态生成当前 Checkpoint 专属的打分日志路径
    case_study_log_path = os.path.join(cfg.output_dir, f"slake_case_study_scored_{cp_name}.txt")
    os.makedirs(os.path.dirname(case_study_log_path), exist_ok=True)
    if os.path.exists(case_study_log_path):
        os.remove(case_study_log_path)

    # 1. 每次都获取一个极其干净的底座
    base_model = loader.load_model()
    model = PeftModel.from_pretrained(base_model, weights_path)

    # =========================================================
    # 2. 手工挂载 MoE 视觉残差架构
    # =========================================================
    vision_tower = model.base_model.model.model.visual
    ref_param = next(vision_tower.parameters())
    adapter_dtype = ref_param.dtype if ref_param.is_floating_point() else torch.bfloat16

    adapter_global = VisualAdapter_Global(hidden_dim=cfg.visual_adapter_hidden_dim, r=cfg.visual_adapter_r,
                                          kernel_size=cfg.global_adapter_kernel_size)
    adapter_local = VisualAdapter_Local(hidden_dim=cfg.visual_adapter_hidden_dim, r=cfg.visual_adapter_r,
                                        kernel_size=cfg.local_adapter_kernel_size)
    adapter_region = VisualAdapter_Region(hidden_dim=cfg.visual_adapter_hidden_dim, r=cfg.visual_adapter_r,
                                          kernel_size=cfg.region_adapter_kernel_size)

    if cfg.router_mode == "dynamic":
        fusion_layer = Qwen3VLMoEVisualAdapterDynamicFusion(
            hidden_dim=cfg.visual_adapter_hidden_dim,
            adapter_global=adapter_global,
            adapter_local=adapter_local,
            adapter_region=adapter_region,
            moe_alpha=cfg.moe_alpha,
        )
        model.biomed_extractor = biomed_extractor.to(device=ref_param.device, dtype=adapter_dtype)
    else:
        fusion_layer = Qwen3VLMoEVisualAdapterFixedFusion(
            hidden_dim=cfg.visual_adapter_hidden_dim,
            adapter_global=adapter_global,
            adapter_local=adapter_local,
            adapter_region=adapter_region,
            fixed_weights=cfg.fixed_weights,
            moe_alpha=cfg.moe_alpha,
        )

    vision_tower.res_adapter = fusion_layer.to(device=ref_param.device, dtype=adapter_dtype)

    adapter_pt_path = os.path.join(weights_path, "visual_adapter.pt")
    if not os.path.exists(adapter_pt_path):
        print(f"⚠️ 跳过: 找不到 Adapter 权重 {adapter_pt_path}")
        del model
        del base_model
        torch.cuda.empty_cache()
        return

    vision_tower.res_adapter.load_state_dict(torch.load(adapter_pt_path, map_location=ref_param.device))
    print(f"✅ 成功接驳 Stage 2 多尺度 MoE 视觉适配器！融合模式: {cfg.router_mode.upper()}")

    # =========================================================
    # 3. 内层劫持 Forward
    # =========================================================
    vision_tower.original_forward = vision_tower.forward

    def patched_vision_forward(self, *args, **kwargs):
        outputs = self.original_forward(*args, **kwargs)
        img_f = getattr(self, "current_biomed_img_feat", None)
        txt_f = getattr(self, "current_biomed_txt_feat", None)
        grid_thw = kwargs.get("grid_thw", None)
        if grid_thw is None and len(args) > 1: grid_thw = args[1]
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

    # =========================================================
    # 🌟 Phase 1: 本地推理与路由权重精准打桩捕获
    # =========================================================
    all_records = []

    with torch.no_grad():
        for batch_inputs, metadata_list in tqdm(test_loader, desc=f"Local Inference ({cp_name})"):
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch_inputs.items()}

            biomed_img = inputs.pop("biomed_image_tensors", None)
            biomed_txt = inputs.pop("biomed_text_tokens", None)

            if cfg.router_mode == "dynamic" and biomed_img is not None and biomed_txt is not None:
                biomed_img = biomed_img.to(dtype=adapter_dtype)
                img_f, txt_f = model.biomed_extractor(biomed_img, biomed_txt)
                vision_tower.current_biomed_img_feat = img_f
                vision_tower.current_biomed_txt_feat = txt_f
            else:
                vision_tower.current_biomed_img_feat = None
                vision_tower.current_biomed_txt_feat = None

            generated_ids = model.generate(**inputs, max_new_tokens=cfg.max_new_tokens, do_sample=cfg.do_sample,
                                           temperature=cfg.temperature)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in
                                     zip(inputs["input_ids"], generated_ids)]
            output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)

            # 💡 强行拦截并存储这一 Batch 的所有动态路由分流权重 (转为float32彻底干掉numpy报错)
            saved_weights = vision_tower.res_adapter.latest_routing_weights.float().cpu().numpy()

            for i, raw_pred in enumerate(output_texts):
                meta = metadata_list[i]
                gt_raw = meta.get("gt_answer", meta.get("answer", ""))
                ans_type = meta.get("answer_type", "").strip().upper()
                is_closed_question = (ans_type == "CLOSED")

                pred_norm = slake_answer_eval_cleaning(raw_pred)
                gt_norm = slake_answer_eval_cleaning(gt_raw)
                is_norm = (pred_norm == gt_norm)

                # 剥离出当前独立样本的 3 项专家权重
                w_g, w_l, w_r = saved_weights[i][0], saved_weights[i][1], saved_weights[i][2]

                all_records.append({
                    "img_id": meta.get('img_id', meta.get('id', meta.get('image_name', 'UNKNOWN'))),
                    "image_path": meta.get('image_path', meta.get('image_name', 'UNKNOWN')),
                    "question": meta["question"],
                    "gt_raw": gt_raw,
                    "gt_norm": gt_norm,
                    "pred_raw": raw_pred.strip(),
                    "pred_norm": pred_norm,
                    "is_norm_match": is_norm,
                    "question_category": "closed" if is_closed_question else "open",
                    "routing_weights": (w_g, w_l, w_r)
                })

    # 推理完当场释放当前模型的显存，为后面调用大模型裁判留出充足的生路
    del model
    del base_model
    torch.cuda.empty_cache()
    gc.collect()

    # =========================================================
    # 🌟 Phase 2: 完全复刻对齐的 LLM 裁判打分与日志写盘
    # =========================================================
    print(f"⏳ 开始同步进行 LLM 裁判审阅与分流日志写盘...")

    with open(case_study_log_path, "w", encoding="utf-8") as f_log:
        for record in tqdm(all_records, desc=f"LLM Judging & Logging ({cp_name})"):
            w_g, w_l, w_r = record["routing_weights"]

            # 复刻原本的打分划分逻辑
            if record["is_norm_match"]:
                score_mark = "✅ 1 (Correct)"
            elif record["question_category"] == "closed":
                score_mark = "❌ 0 (Wrong)"
            else:
                # 针对不匹配的 Open 题型，精准调用你们库里的 .ask 方法与 5 维入参！
                try:
                    user_prompt = build_llm_judge_user_prompt(
                        question=record["question"],
                        gt_raw=record["gt_raw"],
                        gt_norm=record["gt_norm"],
                        pred_raw=record["pred_raw"],
                        pred_norm=record["pred_norm"]
                    )
                    # 🚀 采用你们独家注册的 ask 函数与系统 Prompt 注入
                    raw_response = llm_client.ask(llm_cfg.vqa_rad_llm_judge_system_prompt, user_prompt, temperature=0.0)
                    parsed_result = parse_llm_judge_response(raw_response)

                    if parsed_result["score"] == "correct":
                        score_mark = "✅ 1 (Correct)"
                    elif parsed_result["score"] == "partially_correct":
                        score_mark = "🔶 0.5 (Partially Correct)"
                    else:
                        score_mark = "❌ 0 (Wrong)"
                except Exception as e:
                    print(f"裁判打分出现意外错误: {e}")
                    score_mark = "⚠️ Error"

            # 📝 客观、绝无主观导向的纯机读式定性分析格式写盘
            f_log.write(f"=== Case ID: {record['img_id']} ===\n")
            f_log.write(f"Image Path: {record['image_path']}\n")
            f_log.write(f"Question: {record['question']}\n")
            f_log.write(f"Ground Truth: {record['gt_raw']}\n")
            f_log.write(f"RoMA Prediction: {record['pred_raw']}\n")
            f_log.write(f"LLM Judge Score: {score_mark}\n")
            f_log.write(
                f"Routing Distribution -> Global(k=1): {w_g:.4f} | Local(k=3): {w_l:.4f} | Region(k=5): {w_r:.4f}\n")
            f_log.write("-" * 60 + "\n")

    print(f"🏆 Checkpoint {cp_name} 寻宝日志安全封存完成！路径 👉 {case_study_log_path}")


def main():
    cfg = Stage2EvalConfig()

    # 自动检索提取父级目录下的所有节点
    base_weight_dir = os.path.dirname(cfg.stage2_weights_dir)
    checkpoint_dirs = glob.glob(os.path.join(base_weight_dir, "checkpoint-*"))
    checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]))

    final_weights_path = os.path.join(base_weight_dir, "final_weights")
    if os.path.exists(final_weights_path):
        checkpoint_dirs.append(final_weights_path)

    if not checkpoint_dirs:
        print(f"❌ 警告: 在 {base_weight_dir} 下未找到任何训练权重！")
        return

    print(
        f"🔍 [全自动多节点定性寻宝启动] 评测模式: [MoE {cfg.router_mode.upper()}] | 共检测到 {len(checkpoint_dirs)} 个节点")

    # 全局初始化 Qwen 处理器与加载器配置
    loader = Qwen3VLQuantizedLoader(
        model_path=cfg.model_name_or_path, processor_path=cfg.model_name_or_path,
        load_in_4bit=cfg.load_in_4bit, bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant, bnb_4bit_compute_dtype=cfg.bnb_4bit_compute_dtype,
        torch_dtype=cfg.torch_dtype, attn_implementation=cfg.attn_implementation, device_map="auto"
    )
    processor = loader.load_processor()
    processor.tokenizer.padding_side = "left"

    # 根据动态模式条件加载 BioMedCLIP 路由端
    biomed_extractor, biomed_transform, biomed_tokenizer = None, None, None
    if cfg.router_mode == "dynamic":
        biomed_extractor, biomed_transform, biomed_tokenizer = load_biomedclip(cfg.biomedclip_path)

    # 挂载数据集与整理流
    test_dataset = SLAKEDataset(json_path=cfg.slake_test_json_path, image_root=cfg.slake_image_root)
    eval_collator = SLAKEEvalCollator(processor, cfg, biomed_transform, biomed_tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=cfg.per_device_eval_batch_size, collate_fn=eval_collator,
                             num_workers=cfg.dataloader_num_workers, shuffle=False)

    # 初始化 LLM 裁判配置
    llm_cfg = LLMAPIConfig()
    llm_client = GPT5MiniClient(api_key=llm_cfg.gpt_5_mini_key, base_url=llm_cfg.base_url,
                                model=llm_cfg.judge_model_name)

    # 大循环逐个遍历并吐出每个 Checkpoint 的专属日志报告
    for cp_path in checkpoint_dirs:
        process_single_checkpoint_to_log(
            weights_path=cp_path,
            loader=loader,
            processor=processor,
            cfg=cfg,
            test_loader=test_loader,
            llm_client=llm_client,
            llm_cfg=llm_cfg,
            biomed_extractor=biomed_extractor
        )

    print(f"\n🏆 全部 {len(checkpoint_dirs)} 个节点的带路由分布打分日志已全部洗干净，安全封存于 👉 {cfg.output_dir}")


if __name__ == "__main__":
    main()