import os
from dataclasses import dataclass, field


@dataclass
class Stage2TrainConfig:
    """阶段二 (Stage 2: VQA-MED-2019) 降维打击全局配置类"""

    # --- 1. 基础与输出路径配置 ---
    # 输出目录 (Stage 2 VQA-MED 专属结果目录)
    output_dir_with_visual_adapter_dynamic: str = "/home/yuqing/Models/RouterB_Plus_MoA/Stage2_VQA_MED_2019/dynamic"
    output_dir_with_visual_adapter_fixed: str = "/home/yuqing/Models/RouterB_Plus_MoA/Stage2_VQA_MED_2019/fixed"

    # ⚠️ 继承 Stage 1 权重的根目录 (基座保持不变)
    stage1_output_dir_with_visual_adapter_dynamic: str = "/home/yuqing/Models/RouterB_Plus_MoA/with_visual_adapter_dynamic"
    stage1_output_dir_with_visual_adapter_fixed: str = "/home/yuqing/Models/RouterB_Plus_MoA/with_visual_adapter_fixed"

    print_rank: int = 0
    seed: int = 2048 #1024 #2048 #2048 #1024 #1912

    # --- 2. VQA-MED-2019 数据集配置 ---
    # 🎯 指向我们刚才测通的 QAPairsByCategory 文件夹
    vqa_med_2019_train_data_path: str = "/home/yuqing/Datas/VQA-Med-2019/train/ImageClef-2019-VQA-Med-Training/QAPairsByCategory"
    vqa_med_2019_train_image_root: str = "/home/yuqing/Datas/VQA-Med-2019/train/ImageClef-2019-VQA-Med-Training/Train_images"

    vqa_med_2019_val_data_path: str = "/home/yuqing/Datas/VQA-Med-2019/val/ImageClef-2019-VQA-Med-Validation/QAPairsByCategory"
    vqa_med_2019_val_image_root: str = "/home/yuqing/Datas/VQA-Med-2019/val/ImageClef-2019-VQA-Med-Validation/Val_images"

    vqa_med_2019_max_size: int = 1024

    # 🚀 降维打击核心：复用极其成功的简答指令 Suffix
    vqa_med_2019_instruction_suffix: str = (
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

    # BioMedCLIP 本地绝对路径
    biomedclip_path: str = "/home/yuqing/Models/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    biomedclip_model_name: str = "ViT-B-16"

    # =========================================================
    # ✨ 核心创新点：视觉端 (Vision) 的残差适配器
    # =========================================================
    visual_adapter_hidden_dim: int = 4096
    visual_adapter_r: int = 16

    router_mode: str = "dynamic" #"dynamic" #"dynamic"  # 切换为 fixed 或 dynamic
    global_adapter_kernel_size: int = 1
    local_adapter_kernel_size: int = 3
    region_adapter_kernel_size: int = 5

    fixed_weights: list[float] = field(default_factory=lambda: [0.333, 0.333, 0.334])

    # 🚀 你的 8 卡消融核心变量
    moe_alpha: float = 0.9 #1 #0.9 #0.7 #0.6 #0.5 #0.4 #0.7 #0.6 #0.5 #0.4 #0.3 #0.1 #0

    # --- 4. LoRA 配置 ---
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # --- 5. 训练超参数 ---
    # ⚠️ VQA-MED 的训练集近 1.3w 条，与 SLAKE 规模相当，这里沿用 SLAKE 的最佳设定
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    num_train_epochs: float = 3.0
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"

    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    logging_steps: int = 10

    # ⚠️ 1.3w 数据量，500步一存是个好选择
    save_steps: int = 500
    save_total_limit: int = 500

    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 8

    def __post_init__(self):
        if self.attn_implementation == "flash_attention_2" and self.torch_dtype != "bfloat16":
            print("⚠️ Warning: flash_attention_2 is best paired with bfloat16!")

        alpha_str = f"{self.moe_alpha:g}"

        if self.router_mode == "dynamic":
            base_output_dir = self.output_dir_with_visual_adapter_dynamic
            base_stage1_dir = self.stage1_output_dir_with_visual_adapter_dynamic
            run_name = f"dynamic_Alpha_{alpha_str}_seed_{self.seed}"
        elif self.router_mode == "fixed":
            base_output_dir = self.output_dir_with_visual_adapter_fixed
            base_stage1_dir = self.stage1_output_dir_with_visual_adapter_fixed
            run_name = f"fixed_Alpha_{alpha_str}_seed_{self.seed}"
        else:
            raise ValueError(f"❌ 不支持的 router_mode: {self.router_mode}")

        # Stage-2 输出目录：和 SLAKE 保持一致
        self.output_dir = f"{base_output_dir}_Alpha_{alpha_str}_seed_{self.seed}"

        # Stage-1 读取目录：优先读取 seed-aware 版本
        stage1_alpha_dir = f"{base_stage1_dir}_Alpha_{alpha_str}_seed_{self.seed}"
        self.stage1_weights_dir = os.path.join(stage1_alpha_dir, "final_weights")

        print("\n" + "=" * 60)
        print(
            f"⚙️ [Stage 2 Train Config] VQA-MED-2019 | 模式: {self.router_mode.upper()} | "
            f"Alpha: {self.moe_alpha} | Seed: {self.seed}"
        )
        print(f"📂 读取 Stage 1 权重: {self.stage1_weights_dir}")
        print(f"💾 训练结果输出目录: {self.output_dir}")
        print("=" * 60 + "\n")