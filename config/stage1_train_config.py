from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    """阶段 (Train) 全局配置类"""

    # --- 1. 基础配置 ---
    output_dir_with_visual_adapter: str = "/home/yuqing/Models/RouterB_Plus/with_visual_adapter"
    output_dir_lora_only_baseline: str = "/home/yuqing/Models/RouterB_Plus/lora_only_baseline"
    print_rank: int = 0
    seed: int = 1912

    # --- 2. 数据集配置 ---
    # -----------------------------
    # MIMIC-CXR dataset config
    # -----------------------------
    mimic_cxr_root: str = "/home/yuqing/Datas/mimic-cxr-jpg-2.1.0"
    mimic_cxr_metadata_csv: str = "/home/yuqing/Datas/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-metadata.csv.gz"
    mimic_cxr_image_root: str = "/home/yuqing/Datas/mimic-cxr-jpg-2.1.0/files"
    mimic_cxr_report_root: str = "/home/yuqing/Datas/mimic-cxr-jpg-2.1.0/reports"
    mimic_cxr_target_section: str = "impression"   # 可改为 "findings" 或 "full_report",也决定我们清洗时提取哪段文本
    mimic_cxr_load_report: bool = True
    mimic_cxr_view_positions: Optional[list[str]] = None
    mimic_cxr_drop_empty_target: bool = True
    # 缓存路径
    mimic_cxr_cache_dir = "/home/yuqing/Datas/mimic-cxr-jpg-2.1.0/cache"
    mimic_cxr_use_indices_cache = True
    mimic_cxr_rebuild_indices_cache = False
    mimic_cxr_cache_prefix="mimic_cxr"
    # 【新增】MIMIC-CXR 专属的全局指令
    mimic_cxr_instruction_suffix: str =("Act as an expert radiologist. "
                                 "Carefully analyze this chest radiograph and provide a comprehensive clinical interpretation.")

    # 为 mimic_cxr 指定 max_size，保持队列整齐
    mimic_cxr_max_size: int = 1024

    # 是否截断数据集用于快速调试 (Smoke Test)。如果不截断，保持为 None 即可跑全量数据。
    max_mimic_cxr_train_samples: Optional[int] = None


    # --- 3. 模型与量化配置 ---
    model_name_or_path: str = "/home/yuqing/Models/Qwen3-VL-8B-Instruct"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"

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

    # --- 5. 训练超参数 ---
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 2
    """
    为什么要在 Wrapper 里开 (True)？ 因为我们需要在底层的 PyTorch 模型层面通过 prepare_model_for_kbit_training 显式开启梯度检查点，并且传入 use_reentrant=False，这是解决内存溢出和计算图报错的基石。
    为什么要在 Trainer 里关 (False)？ 如果在 TrainingArguments 里设为 True，Hugging Face 的 Trainer 会再对模型包裹一层自己的 gradient checkpointing 逻辑。这层逻辑在遇到 DDP 时，极大概率会触发 find_unused_parameters 相关的恐怖报错（计算图断裂）。
    """
    gradient_checkpointing: bool = True


    dataloader_num_workers: int = 12

    # =========================================================
    # ✨ 核心创新点：视觉端 (Vision) 的残差适配器
    # =========================================================
    use_visual_adapter: bool = True #跑仅lora时候False
    visual_adapter_hidden_dim: int = 4096  # Qwen3-VL-8B 探测出的真实视觉-语言对齐维度
    visual_adapter_r: int = 16
    visual_adapter_alpha: float = 0.1


    def __post_init__(self):
        if self.attn_implementation == "flash_attention_2" and self.torch_dtype != "bfloat16":
            print("⚠️ Warning: flash_attention_2 is best paired with bfloat16!")
