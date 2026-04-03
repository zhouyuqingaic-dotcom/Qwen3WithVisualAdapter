import os
import types
import random
import numpy as np
import torch
import torch.distributed as dist
from transformers import Trainer, TrainingArguments
from peft import PeftModel, prepare_model_for_kbit_training
from config.stage2_train_config import Stage2TrainConfig
from datas.vqa_rad_datasets import VQARADDataset
from utils.data_tools.collator.vqa_rad_datasets_train_collator import VQARADTrainCollator
from utils.qwen3vl.qwen3_vl_8B_quant_loader import Qwen3VLQuantizedLoader
from utils.qwen3vl.qwen3_vl_8B_visual_adapter import Qwen3VLVisualAdapter
from utils.ddp.ddp_utils import ddp_print
from transformers import TrainerCallback


class VisualAdapterSaveCallback(TrainerCallback):
    """
    专属回调：在 HF Trainer 自动保存 checkpoint 时，强制将 Visual Adapter 权重一并保存！
    支持模式感知：纯 LoRA 模式下自动静默跳过。
    """

    def __init__(self, use_visual_adapter: bool):
        self.use_visual_adapter = use_visual_adapter

    def on_save(self, args, state, control, **kwargs):
        # 🚨 如果是纯 LoRA 模式，直接优雅跳过，绝不报错！
        if not self.use_visual_adapter:
            return

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
            print(f"\n  [Callback] ✨ 视觉适配器权重已同步保存至: {adapter_save_path}")
        except Exception as e:
            print(f"\n  [Callback] ❌ 保存视觉适配器失败: {e}")

def set_seed(seed: int):
    """
    全局随机种子设置，保证多卡 DDP 环境下每次初始化的权重和数据采样顺序一致。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 保证 cuDNN 算子确定性（可能略微影响训练速度，但为了科研复现是值得的）
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
    use_visual_adapter = cfg.use_visual_adapter

    # 动态路由输出目录与继承路径
    if use_visual_adapter:
        output_dir = cfg.output_dir_with_visual_adapter
        stage1_weights_dir = cfg.stage1_weights_with_visual_adapter
    else:
        output_dir = cfg.output_dir_lora_only_baseline
        stage1_weights_dir = cfg.stage1_weights_lora_only_baseline

    set_seed(cfg.seed)

    ddp_print("\n" + "=" * 60, print_rank=cfg.print_rank)
    ddp_print(f"🚀 [1/6] 启动 Route B+ (Stage 2: VQA-RAD) 分布式训练！", print_rank=cfg.print_rank)
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

    # ==========================================
    # 3. 🎯 继承 Stage 1 权重并唤醒梯度 (极度核心)
    # ==========================================
    ddp_print("\n⏳ [4/6] 正在装载 Stage 1 权重并执行脑机接驳...", print_rank=cfg.print_rank)

    # 3.1 开启基础的梯度检查点
    model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    if hasattr(model, "config"):
        model.config.use_cache = False

    # 3.2 挂载 Stage 1 的 LoRA，并强制开启 is_trainable=True 使其可被继续微调！
    peft_model = PeftModel.from_pretrained(model, stage1_weights_dir, is_trainable=True)

    if hasattr(peft_model, "enable_input_require_grads"):
        peft_model.enable_input_require_grads()

    # 3.3 继承并挂载 Visual Adapter
    if use_visual_adapter:
        vision_tower = peft_model.base_model.model.model.visual
        ref_param = next(vision_tower.parameters())
        adapter_dtype = ref_param.dtype if ref_param.is_floating_point() else torch.bfloat16

        # 实例化 Adapter
        adapter = Qwen3VLVisualAdapter(
            hidden_dim=cfg.visual_adapter_hidden_dim,
            r=cfg.visual_adapter_r,
            init_alpha=cfg.visual_adapter_alpha,
        ).to(device=ref_param.device, dtype=adapter_dtype)

        # ✨ 加载 Stage 1 保存的物理参数
        adapter_pt_path = os.path.join(stage1_weights_dir, "visual_adapter.pt")
        if not os.path.exists(adapter_pt_path):
            raise FileNotFoundError(f"开启了视觉适配器，但在 Stage 1 目录中找不到 {adapter_pt_path}！")

        adapter.load_state_dict(torch.load(adapter_pt_path, map_location=ref_param.device))
        adapter.requires_grad_(True)  # 保证在 Stage 2 中继续学习！

        # 上户口
        vision_tower.res_adapter = adapter
        vision_tower.original_forward = vision_tower.forward

        # 脑机接驳
        def patched_forward(self, *args, **kwargs):
            outputs = self.original_forward(*args, **kwargs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                outputs.pooler_output = self.res_adapter(outputs.pooler_output)
            if hasattr(outputs, "deepstack_features") and outputs.deepstack_features is not None:
                outputs.deepstack_features = [self.res_adapter(x) for x in outputs.deepstack_features]
            return outputs

        vision_tower.forward = types.MethodType(patched_forward, vision_tower)
        ddp_print("✅ Visual Adapter 权重装载成功，已准备好吸收 VQA-RAD 知识！", print_rank=cfg.print_rank)

    peft_model.print_trainable_parameters()

    # ==========================================
    # 4. 初始化 Collator
    # ==========================================
    ddp_print("\n⏳ [5/6] 挂载 VQA-RAD 专属 Collator...", print_rank=cfg.print_rank)
    collator = VQARADTrainCollator(processor, cfg)

    # ==========================================
    # 5. 配置 HF TrainingArguments & 启动
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
        # 把大开关的状态传给保安，让他知道该不该查岗
        callbacks=[VisualAdapterSaveCallback(use_visual_adapter=use_visual_adapter)], #需要教 Hugging Face 的 Trainer 在自动保存时，顺手把我们的 Visual Adapter 也存下来
    )

    trainer.train()

    # ==========================================
    # 6. 保存最终权重 (Stage 2 结业)
    # ==========================================
    if local_rank in [-1, 0]:
        final_save_path = os.path.join(output_dir, "final_weights")
        os.makedirs(final_save_path, exist_ok=True)

        trainer.save_model(final_save_path)
        processor.save_pretrained(final_save_path)
        ddp_print(f"\n🎉 Stage 2 训练完成！LoRA 已保存至: {final_save_path}", print_rank=cfg.print_rank)

        if use_visual_adapter:
            try:
                adapter_module = peft_model.base_model.model.model.visual.res_adapter
                adapter_save_path = os.path.join(final_save_path, "visual_adapter.pt")
                torch.save(adapter_module.state_dict(), adapter_save_path)
                ddp_print(f"✨ 视觉残差适配器 (Stage 2版) 权重已单独安全保存至: {adapter_save_path}",
                          print_rank=cfg.print_rank)
            except Exception as e:
                ddp_print(f"❌ 保存 Visual Adapter 权重时发生错误: {e}", print_rank=cfg.print_rank)

    # 释放资源
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()