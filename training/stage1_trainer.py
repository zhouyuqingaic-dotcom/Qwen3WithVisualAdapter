import os
import random
import numpy as np
import torch
import torch.distributed as dist
from transformers import Trainer, TrainingArguments

# 1. 导入配置
from config.stage1_train_config import TrainConfig

# 2. 导入数据集与 Collator
from datas.mimic_cxr_datasets import MIMICCXRDataset
from utils.data_tools.collator.mimic_cxr_datasets_train_collator import MIMICCXRTrainCollator

# 3. 导入模型加载与包装器
from utils.qwen3vl.qwen3_vl_8B_quant_loader import Qwen3VLQuantizedLoader
# ⚠️ 注意：已经删除了旧版的 Qwen3VLLoraWrapper 导入
from utils.qwen3vl.qwen3_vl_8B_lora_wrapper import Qwen3VLLoraAndVisualAdapterWrapper
from utils.biomedclip.biomed_clip_loader import load_biomedclip

# 4. 导入 DDP 打印工具
from utils.ddp.ddp_utils import ddp_print


def set_seed(seed: int):
    """
    全局随机种子设置，保证多卡 DDP 环境下每次初始化的权重和数据采样顺序一致。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 保证 cuDNN 算子确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # ==========================================
    # 0. DDP 环境感知与初始化
    # ==========================================
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        # 强制当前进程只使用自己分配到的那张卡，防止显存抢占
        torch.cuda.set_device(local_rank)

    cfg = TrainConfig()

    # 【更新】：直接使用 Config 自动拼接好的输出目录 (跟随 router_mode 变化)
    #  直接向 Config 索取最终的路径！不再做任何判断！
    output_dir = cfg.output_dir

    # 锁定全局随机种子
    set_seed(cfg.seed)

    ddp_print("\n" + "=" * 60, print_rank=cfg.print_rank)
    ddp_print(f"🚀 [1/6] 启动 Route B+ (Stage 1) 分布式训练！当前模式: {cfg.router_mode.upper()}",
              print_rank=cfg.print_rank)
    ddp_print("=" * 60, print_rank=cfg.print_rank)

    # 主进程负责创建输出目录
    if local_rank in [-1, 0]:
        os.makedirs(output_dir, exist_ok=True)

    # ==========================================
    # 1. 加载 MIMIC-CXR 数据集
    # ==========================================
    ddp_print("⏳ [2/6] 正在加载 MIMIC-CXR 训练集 (基于缓存)...", print_rank=cfg.print_rank)
    train_dataset = MIMICCXRDataset(
        csv_path=cfg.mimic_cxr_metadata_csv,
        image_root=cfg.mimic_cxr_image_root,
        report_root=cfg.mimic_cxr_report_root,
        target_section=cfg.mimic_cxr_target_section,
        load_report=cfg.mimic_cxr_load_report,
        allowed_view_positions=cfg.mimic_cxr_view_positions,
        drop_empty_target=cfg.mimic_cxr_drop_empty_target,
        cache_dir=cfg.mimic_cxr_cache_dir,
        use_indices_cache=cfg.mimic_cxr_use_indices_cache,
        rebuild_indices_cache=cfg.mimic_cxr_rebuild_indices_cache,
        cache_prefix=cfg.mimic_cxr_cache_prefix,
    )
    if hasattr(cfg, "max_mimic_cxr_train_samples") and cfg.max_mimic_cxr_train_samples is not None:
        train_dataset.samples = train_dataset.samples[:cfg.max_mimic_cxr_train_samples]
        ddp_print(f"⚠️ 已截断数据集用于调试，当前样本量: {len(train_dataset)}", print_rank=cfg.print_rank)
    else:
        ddp_print(f"✅ 数据集加载完成，共有 {len(train_dataset)} 条训练样本。", print_rank=cfg.print_rank)

    # ==========================================
    # 2. 加载 Qwen3-VL 4-bit 底座模型与 Processor
    # ==========================================
    ddp_print("\n⏳ [3/6] 正在加载 Qwen3-VL 4-bit 底座模型...", print_rank=cfg.print_rank)
    loader = Qwen3VLQuantizedLoader(
        model_path=cfg.model_name_or_path,
        processor_path=cfg.model_name_or_path,
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=cfg.bnb_4bit_compute_dtype,
        torch_dtype=cfg.torch_dtype,
        attn_implementation=cfg.attn_implementation,
        device_map={"": local_rank} if local_rank != -1 else "auto",
    )
    base_model, processor = loader.load()

    processor.tokenizer.padding_side = "right"
    ddp_print("✅ 底座加载完毕！", print_rank=cfg.print_rank)

    # ==========================================
    # 🌟 修复：全局单例初始化 BioMedCLIP 引擎 (基于注册表注入)
    # ==========================================
    biomed_extractor = None
    biomed_transform = None
    biomed_tokenizer = None

    if cfg.router_mode == "dynamic":
        # 直接调用重构后的 Loader！
        biomed_extractor, biomed_transform, biomed_tokenizer = load_biomedclip(
            biomedclip_path=cfg.biomedclip_path,
            print_rank=cfg.print_rank
        )


    # ==========================================
    # 3. 施加黑魔法：植入 LoRA 与 MoE Visual Adapter
    # ==========================================
    ddp_print("\n⏳ [4/6] 正在执行模型接驳手术 (注入 LoRA 及 MoE Visual Adapter)...", print_rank=cfg.print_rank)

    # 彻底抛弃旧分支，强制挂载完全体 MoE Wrapper
    wrapper = Qwen3VLLoraAndVisualAdapterWrapper(
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        lora_target_modules=cfg.lora_target_modules,
        gradient_checkpointing=cfg.gradient_checkpointing,
        visual_adapter_hidden_dim=cfg.visual_adapter_hidden_dim,
        visual_adapter_r=cfg.visual_adapter_r,
        router_mode=cfg.router_mode,
        biomed_extractor=biomed_extractor,  #  依赖注入：分发大脑给 Wrapper
        global_adapter_kernel_size=cfg.global_adapter_kernel_size,
        local_adapter_kernel_size=cfg.local_adapter_kernel_size,
        region_adapter_kernel_size=cfg.region_adapter_kernel_size,
        fixed_weights=cfg.fixed_weights
    )

    model = wrapper.wrap(base_model)
    ddp_print("✅ 模型接驳完成！", print_rank=cfg.print_rank)

    # ==========================================
    # 4. 初始化 Collator
    # ==========================================
    ddp_print("\n⏳ [5/6] 挂载 MIMIC-CXR 专属 Collator...", print_rank=cfg.print_rank)

    # 👈 依赖注入：分发轻量级手脚给 Collator
    collator = MIMICCXRTrainCollator(
        processor=processor,
        cfg=cfg,
        biomed_transform=biomed_transform,
        biomed_tokenizer=biomed_tokenizer
    )

    # ==========================================
    # 5. 配置 HF TrainingArguments
    # ==========================================
    ddp_print(f"\n🔥 [6/6] 启动 Hugging Face Trainer... 设定轮数: {cfg.num_train_epochs}", print_rank=cfg.print_rank)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup_steps,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=(cfg.torch_dtype == "bfloat16" or cfg.torch_dtype == torch.bfloat16),
        fp16=(cfg.torch_dtype == "float16" or cfg.torch_dtype == torch.float16),
        dataloader_num_workers=cfg.dataloader_num_workers,

        gradient_checkpointing=False,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    # 🚀 正式点火训练
    trainer.train()

    # ==========================================
    # 6. 保存最终权重 (无条件保存 MoE 组件)
    # ==========================================
    if local_rank in [-1, 0]:
        final_save_path = os.path.join(output_dir, "final_weights")
        os.makedirs(final_save_path, exist_ok=True)

        trainer.save_model(final_save_path)
        processor.save_pretrained(final_save_path)
        ddp_print(f"\n🎉 训练完成！LoRA 权重与 Processor 已保存至: {final_save_path}", print_rank=cfg.print_rank)

        # 【修改】：不再检查 use_visual_adapter，无条件安全提取 MoE 权重
        try:
            adapter_module = model.base_model.model.model.visual.res_adapter
            adapter_state_dict = adapter_module.state_dict()

            adapter_save_path = os.path.join(final_save_path, "visual_adapter.pt")
            torch.save(adapter_state_dict, adapter_save_path)
            ddp_print(f"✨ MoE 多尺度残差适配器权重已单独安全保存至: {adapter_save_path}", print_rank=cfg.print_rank)
        except Exception as e:
            ddp_print(f"❌ 保存 Visual Adapter 权重时发生错误: {e}", print_rank=cfg.print_rank)

    # ==========================================
    # 7. 优雅释放 DDP 资源
    # ==========================================
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()