import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Stage2TrainConfig:
    """阶段二 (Stage 2: VQA-RAD) 全局配置类"""

    # --- 1. 基础与输出路径配置 ---
    # 自动将输出目录指向 Stage 2 专属文件夹
    output_dir_with_visual_adapter: str = "/home/yuqing/Models/RouterB_Plus/Stage2_VQA_RAD/with_visual_adapter"
    output_dir_lora_only_baseline: str = "/home/yuqing/Models/RouterB_Plus/Stage2_VQA_RAD/lora_only_baseline"

    # ⚠️ 继承 Stage 1 权重的根目录
    stage1_weights_with_visual_adapter: str = "/home/yuqing/Models/RouterB_Plus/with_visual_adapter/final_weights"
    stage1_weights_lora_only_baseline: str = "/home/yuqing/Models/RouterB_Plus/lora_only_baseline/final_weights"

    print_rank: int = 0
    seed: int = 1912

    # --- 2. VQA-RAD 数据集配置 ---
    vqa_rad_train_jsonl_path: str = "/home/yuqing/Datas/VQA-RAD/train.jsonl"
    vqa_rad_test_jsonl_path: str = "/home/yuqing/Datas/VQA-RAD/test.jsonl"
    vqa_rad_image_root: str = "/home/yuqing/Datas/VQA-RAD/images"
    vqa_rad_max_size: int = 1024

    # 阶段二专属指令
    vqa_rad_instruction_suffix: str = (
        "Answer the question briefly and directly based on the image. Use a short medical term or phrase when possible. "
        "For yes/no questions, answer with yes or no. Do not add unnecessary explanation."
    )

    # --- 3. 模型与量化配置 ---
    model_name_or_path: str = "/home/yuqing/Models/Qwen3-VL-8B-Instruct"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"

    # --- 4. 视觉 Adapter 维度参数 (如果启用的话，需与 Stage1 保持绝对一致) ---
    # =========================================================
    # ✨ 核心创新点开关：是否启用视觉残差适配器
    # =========================================================
    use_visual_adapter: bool = False  # Ablation 实验时改为 False
    visual_adapter_hidden_dim: int = 4096
    visual_adapter_r: int = 16
    visual_adapter_alpha: float = 0.1

    # --- 5. 训练超参数 (针对 VQA-RAD 小数据集微调) ---
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    num_train_epochs: float = 5.0  # VQA-RAD 数据量小，Epoch 适当拉大
    learning_rate: float = 1e-5  # 学习率比 Stage 1 略低，防止冲刷已有知识
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 50  # 相应缩短 warmup
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 2
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 8

    def __post_init__(self):
        if self.attn_implementation == "flash_attention_2" and self.torch_dtype != "bfloat16":
            print("⚠️ Warning: flash_attention_2 is best paired with bfloat16!")