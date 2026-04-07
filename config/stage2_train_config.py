import os
from dataclasses import dataclass, field
from dataclasses import dataclass


@dataclass
class Stage2TrainConfig:
    """阶段二 (Stage 2: VQA-RAD) 全局配置类"""

    # --- 1. 基础与输出路径配置 ---
    # 输出目录 (Stage 2 结果)
    output_dir_with_visual_adapter_dynamic: str = "/home/yuqing/Models/RouterB_Plus_MoA/Stage2_VQA_RAD/dynamic"
    output_dir_with_visual_adapter_fixed: str = "/home/yuqing/Models/RouterB_Plus_MoA/Stage2_VQA_RAD/fixed"

    # ⚠️ 继承 Stage 1 权重的根目录
    stage1_output_dir_with_visual_adapter_dynamic: str = "/home/yuqing/Models/RouterB_Plus_MoA/with_visual_adapter_dynamic"
    stage1_output_dir_with_visual_adapter_fixed: str = "/home/yuqing/Models/RouterB_Plus_MoA/with_visual_adapter_fixed"

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

    # BioMedCLIP 本地绝对路径 (OpenCLIP 格式)
    biomedclip_path: str = "/home/yuqing/Models/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    # 【新增】：明确指定架构名称，BiomedCLIP 基于 ViT-B-16
    biomedclip_model_name: str = "ViT-B-16"

    # =========================================================
    # ✨ 核心创新点：视觉端 (Vision) 的残差适配器
    # =========================================================
    visual_adapter_hidden_dim: int = 4096  # Qwen3-VL-8B 探测出的真实视觉-语言对齐维度
    visual_adapter_r: int = 16
    # 【新增】多尺度 Visual Adapter 与 Router 专属配置
    # router_mode: str = "dynamic"  # 可选: "dynamic" 或 "fixed"
    router_mode: str = "dynamic"  # 可选: "dynamic" 或 "fixed"
    global_adapter_kernel_size: int = 1
    local_adapter_kernel_size: int = 3
    region_adapter_kernel_size: int = 5
    #  新增：当 router_mode="fixed" 时的硬融合比例
    fixed_weights: list[float] = field(default_factory=lambda: [0.33, 0.33, 0.34])
    
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
    save_steps: int = 200 # 每隔 200 步保存一次
    save_total_limit: int = 500 #上限拉高，保存所有checkpoint
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 8

    def __post_init__(self):
        if self.attn_implementation == "flash_attention_2" and self.torch_dtype != "bfloat16":
            print("⚠️ Warning: flash_attention_2 is best paired with bfloat16!")
        # 🚀 绝杀：根据 router_mode 动态绑定输出路径，并且自动拼接 "final_weights"
        if self.router_mode == "dynamic":
            self.output_dir = self.output_dir_with_visual_adapter_dynamic
            self.stage1_weights_dir = os.path.join(self.stage1_output_dir_with_visual_adapter_dynamic,
                                                   "final_weights")
        elif self.router_mode == "fixed":
            self.output_dir = self.output_dir_with_visual_adapter_fixed
            self.stage1_weights_dir = os.path.join(self.stage1_output_dir_with_visual_adapter_fixed,
                                                   "final_weights")
        else:
            raise ValueError(f"❌ 不支持的 router_mode: {self.router_mode}，只能是 'dynamic' 或 'fixed'")