import os
import random
import numpy as np
import torch
import torch.distributed as dist
from transformers import Trainer, TrainingArguments, TrainerCallback

from config.stage2_train_config import Stage2TrainConfig
from datas.vqa_rad_datasets import VQARADDataset
from utils.data_tools.collator.vqa_rad_datasets_train_collator import VQARADTrainCollator
from utils.qwen3vl.qwen3_vl_8B_quant_loader import Qwen3VLQuantizedLoader

# 🚀 替换为终极完全体 Wrapper 与 BioMedCLIP 加载器
from utils.qwen3vl.qwen3_vl_8B_lora_wrapper import Qwen3VLLoraAndVisualAdapterWrapper
from utils.biomedclip.biomed_clip_loader import load_biomedclip
from utils.ddp.ddp_utils import ddp_print

# 引入 PEFT 的状态字典注入工具
from peft import set_peft_model_state_dict
from safetensors.torch import load_file


class VisualAdapterSaveCallback(TrainerCallback):
    """
    专属回调：在 HF Trainer 自动保存 checkpoint 时，强制将 MoE Visual Adapter 权重一并保存！
    """
    def on_save(self, args, state, control, **kwargs):
        checkpoint_folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        model = kwargs["model"]

        try:
            # 兼容 DDP 和 PEFT 包装的取法
            if hasattr(model, "module"):  # DDP wrapper
                adapter_module = model.module.base_model.model.model.visual.res_adapter
            else:
                adapter_module = model.base_model.model.model.visual.res_adapter

            adapter_save_path = os.path.join(checkpoint_folder, "visual_adapter.pt")
            torch.save(adapter_module.state_dict(), adapter_save_path)
            print(f"\n  [Callback] ✨ MoE 多尺度视觉适配器权重已同步保存至: {adapter_save_path}")
        except Exception as e:
            print(f"\n  [Callback] ❌ 保存 MoE 视觉适配器失败: {e}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # ==========================================
    # 0. DDP 环境感知与初始化
    # ==========================================
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)

    cfg = Stage2TrainConfig()
    output_dir = cfg.output_dir
    stage1_weights_dir = cfg.stage1_weights_dir

    set_seed(cfg.seed)

    ddp_print("\n" + "=" * 60, print_rank=cfg.print_rank)
    ddp_print(f"🚀 [1/6] 启动 Route B+ (Stage 2: VQA-RAD) 分布式微调！模式: {cfg.router_mode.upper()}", print_rank=cfg.print_rank)
    ddp_print(f"🔗 继承 Stage 1 权重目录: {stage1_weights_dir}", print_rank=cfg.print_rank)
    ddp_print("=" * 60, print_rank=cfg.print_rank)

    if local_rank in [-1, 0]:
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(stage1_weights_dir):
            raise FileNotFoundError(f"❌ 找不到 Stage 1 权重，请核实路径: {stage1_weights_dir}")

    # ==========================================
    # 1. 加载 VQA-RAD 数据集
    # ==========================================
    ddp_print("⏳ [2/6] 正在加载 VQA-RAD 训练集...", print_rank=cfg.print_rank)
    train_dataset = VQARADDataset(
        jsonl_path=cfg.vqa_rad_train_jsonl_path,
        image_root=cfg.vqa_rad_image_root,
    )
    ddp_print(f"✅ 数据集加载完成，共有 {len(train_dataset)} 条训练样本。", print_rank=cfg.print_rank)

    # ==========================================
    # 2. 加载 Qwen3-VL 4-bit 底座模型
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
    # 3. 初始化 BioMedCLIP 引擎 (Dynamic 模式专属)
    # ==========================================
    biomed_extractor = None
    biomed_transform = None
    biomed_tokenizer = None

    if cfg.router_mode == "dynamic":
        biomed_extractor, biomed_transform, biomed_tokenizer = load_biomedclip(
            biomedclip_path=cfg.biomedclip_path,
            print_rank=cfg.print_rank
        )

    # ==========================================
    # 4. 🎯 模型接驳与 Stage 1 完美夺舍 (极度核心)
    # ==========================================
    ddp_print("\n⏳ [4/6] 正在执行模型接驳与 Stage 1 权重继承...", print_rank=cfg.print_rank)

    # 4.1 使用 Wrapper 组装架构
    wrapper = Qwen3VLLoraAndVisualAdapterWrapper(
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        lora_target_modules=cfg.lora_target_modules,
        gradient_checkpointing=cfg.gradient_checkpointing,
        visual_adapter_hidden_dim=cfg.visual_adapter_hidden_dim,
        visual_adapter_r=cfg.visual_adapter_r,
        router_mode=cfg.router_mode,
        biomed_extractor=biomed_extractor,
        global_adapter_kernel_size=cfg.global_adapter_kernel_size,
        local_adapter_kernel_size=cfg.local_adapter_kernel_size,
        region_adapter_kernel_size=cfg.region_adapter_kernel_size,
        fixed_weights=cfg.fixed_weights
    )
    peft_model = wrapper.wrap(base_model)

    # 4.2 🚀 绝杀：注入 Stage 1 的 LoRA 权重
    lora_safe_path = os.path.join(stage1_weights_dir, "adapter_model.safetensors")
    lora_bin_path = os.path.join(stage1_weights_dir, "adapter_model.bin")

    if os.path.exists(lora_safe_path):
        set_peft_model_state_dict(peft_model, load_file(lora_safe_path))
        ddp_print(f"✅ 成功接驳 Stage 1 LoRA 权重 (safetensors)", print_rank=cfg.print_rank)
    elif os.path.exists(lora_bin_path):
        set_peft_model_state_dict(peft_model, torch.load(lora_bin_path, map_location="cpu"))
        ddp_print(f"✅ 成功接驳 Stage 1 LoRA 权重 (bin)", print_rank=cfg.print_rank)
    else:
        raise FileNotFoundError(f"❌ 找不到 Stage 1 的 LoRA 权重文件，请检查: {stage1_weights_dir}")

    # 4.3 🚀 注入 Stage 1 的 MoE 视觉适配器权重
    adapter_pt_path = os.path.join(stage1_weights_dir, "visual_adapter.pt")
    if not os.path.exists(adapter_pt_path):
        raise FileNotFoundError(f"❌ 找不到 Stage 1 的 Visual Adapter 权重: {adapter_pt_path}")

    adapter_state_dict = torch.load(adapter_pt_path, map_location="cpu")
    peft_model.base_model.model.model.visual.res_adapter.load_state_dict(adapter_state_dict)
    ddp_print(f"✅ 成功接驳 Stage 1 MoE 多尺度视觉适配器权重！", print_rank=cfg.print_rank)

    peft_model.print_trainable_parameters()

    # ==========================================
    # 5. 初始化 Collator
    # ==========================================
    ddp_print("\n⏳ [5/6] 挂载 VQA-RAD 专属 Collator...", print_rank=cfg.print_rank)
    collator = VQARADTrainCollator(
        processor=processor,
        cfg=cfg,
        biomed_transform=biomed_transform,
        biomed_tokenizer=biomed_tokenizer
    )

    # ==========================================
    # 6. 配置 HF TrainingArguments & 启动
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
        gradient_checkpointing=False,  # 规避 HF 检查点重叠坑
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to="none",
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=[VisualAdapterSaveCallback()],
    )

    trainer.train()

    # ==========================================
    # 7. 保存最终权重 (Stage 2 结业)
    # ==========================================
    if local_rank in [-1, 0]:
        final_save_path = os.path.join(output_dir, "final_weights")
        os.makedirs(final_save_path, exist_ok=True)

        trainer.save_model(final_save_path)
        processor.save_pretrained(final_save_path)
        ddp_print(f"\n🎉 Stage 2 训练完成！LoRA 已保存至: {final_save_path}", print_rank=cfg.print_rank)

        try:
            adapter_module = peft_model.base_model.model.model.visual.res_adapter
            adapter_save_path = os.path.join(final_save_path, "visual_adapter.pt")
            torch.save(adapter_module.state_dict(), adapter_save_path)
            ddp_print(f"✨ 视觉残差适配器 (Stage 2版) 权重已单独安全保存至: {adapter_save_path}", print_rank=cfg.print_rank)
        except Exception as e:
            ddp_print(f"❌ 保存 Visual Adapter 权重时发生错误: {e}", print_rank=cfg.print_rank)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()