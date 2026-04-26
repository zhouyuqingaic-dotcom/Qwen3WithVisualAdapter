import os
from dataclasses import dataclass, field


@dataclass
class Stage2TrainConfig:
    """阶段二 (Stage 2: SLAKE) 降维打击全局配置类"""

    # --- 1. 基础与输出路径配置 ---
    # 输出目录 (Stage 2 SLAKE 专属结果目录)
    output_dir_with_visual_adapter_dynamic: str = "/home/yuqing/Models/RouterB_Plus_MoA/Stage2_SLAKE/dynamic"
    output_dir_with_visual_adapter_fixed: str = "/home/yuqing/Models/RouterB_Plus_MoA/Stage2_SLAKE/fixed"

    # ⚠️ 继承 Stage 1 权重的根目录 (保持不变，因为基座都是同一个 MIMIC-CXR 权重)
    stage1_output_dir_with_visual_adapter_dynamic: str = "/home/yuqing/Models/RouterB_Plus_MoA/with_visual_adapter_dynamic"
    stage1_output_dir_with_visual_adapter_fixed: str = "/home/yuqing/Models/RouterB_Plus_MoA/with_visual_adapter_fixed"

    print_rank: int = 0
    seed: int = 1912

    # --- 2. SLAKE 数据集配置 ---
    slake_train_json_path: str = "/home/yuqing/Datas/SLAKE/Slake1.0/train.json"
    slake_val_json_path: str = "/home/yuqing/Datas/SLAKE/Slake1.0/validate.json"
    slake_test_json_path: str = "/home/yuqing/Datas/SLAKE/Slake1.0/test.json"
    slake_image_root: str = "/home/yuqing/Datas/SLAKE/Slake1.0/imgs"
    slake_max_size: int = 1024

    # 🚀 降维打击核心：无缝复用 VQA-RAD 极其成功的 Suffix
    slake_instruction_suffix: str = (
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

    # BioMedCLIP 本地绝对路径 (OpenCLIP 格式)
    biomedclip_path: str = "/home/yuqing/Models/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    biomedclip_model_name: str = "ViT-B-16"

    # =========================================================
    # ✨ 核心创新点：视觉端 (Vision) 的残差适配器
    # =========================================================
    visual_adapter_hidden_dim: int = 4096  # Qwen3-VL-8B 探测出的真实视觉-语言对齐维度
    visual_adapter_r: int = 16

    router_mode: str = "dynamic"  # 保持动态路由
    global_adapter_kernel_size: int = 1
    local_adapter_kernel_size: int = 3
    region_adapter_kernel_size: int = 5

    fixed_weights: list[float] = field(default_factory=lambda: [0.33, 0.33, 0.34])

    # 🚀 【核心锁死】直接将 VQA-RAD 上推导出的通用物理边界 迁移过来！
    moe_alpha: float = 1 #0.9

    # --- 4. LoRA 配置 ---
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # --- 5. 训练超参数 (针对 SLAKE 万级数据微调) ---
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2

    # ⚠️ SLAKE 数据量(1.2w)是 VQA-RAD 的 4 倍，Epoch 需要降低，防止过拟合
    num_train_epochs: float = 3.0
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"

    # ⚠️ Warmup 相应增加，适应更长的总 Step
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    logging_steps: int = 10

    # ⚠️ Step 变多了，拉长保存间隔，防止硬盘爆炸
    save_steps: int = 500
    save_total_limit: int = 500  # 保留所有检查点，方便我们后续画曲线

    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 8

    def __post_init__(self):
        if self.attn_implementation == "flash_attention_2" and self.torch_dtype != "bfloat16":
            print("⚠️ Warning: flash_attention_2 is best paired with bfloat16!")

        # 🚀 1. 根据 router_mode 提取基础路径 (千万不要在这里提前拼接 final_weights)
        if self.router_mode == "dynamic":
            base_output_dir = self.output_dir_with_visual_adapter_dynamic
            base_stage1_dir = self.stage1_output_dir_with_visual_adapter_dynamic
        elif self.router_mode == "fixed":
            base_output_dir = self.output_dir_with_visual_adapter_fixed
            base_stage1_dir = self.stage1_output_dir_with_visual_adapter_fixed
        else:
            raise ValueError(f"❌ 不支持的 router_mode: {self.router_mode}，只能是 'dynamic' 或 'fixed'")

        # 🚀 2. 给 Stage 2 的输出路径动态追加 Alpha 后缀
        self.output_dir = f"{base_output_dir}_Alpha_{self.moe_alpha}"

        # 🚀 3. 给 Stage 1 的读取路径追加 Alpha 后缀，然后再在最末端拼接 "final_weights"
        stage1_alpha_dir = f"{base_stage1_dir}_Alpha_{self.moe_alpha}"
        self.stage1_weights_dir = os.path.join(stage1_alpha_dir, "final_weights")