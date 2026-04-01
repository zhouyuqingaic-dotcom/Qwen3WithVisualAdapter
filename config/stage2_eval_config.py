import os
from dataclasses import dataclass

@dataclass
class Stage2EvalConfig:
    """Stage 2 (VQA-RAD) 终极评测配置类 (支持双臂网络切换)"""

    # =========================================================
    # ✨ 核心创新点开关：是否启用视觉残差适配器 (用于切换评测目标)
    # =========================================================
    use_visual_adapter: bool = True  # Ablation 实验时改为 False

    # 自动定位对应的 Stage 2 权重路径
    stage2_weights_with_adapter: str = "/home/yuqing/Models/RouterB_Plus/Stage2_VQA_RAD/with_visual_adapter/final_weights"
    stage2_weights_baseline: str = "/home/yuqing/Models/RouterB_Plus/Stage2_VQA_RAD/lora_only_baseline/final_weights"

    # --- 1. 任务协议与 Prompt ---
    vqa_rad_instruction_suffix: str = (
        "Answer the question briefly and directly based on the image. Use a short medical term or phrase when possible. "
        "For yes/no questions, answer with yes or no. Do not add unnecessary explanation."
    )

    # --- 2. 数据集配置 ---
    # vqa_rad_test_jsonl_path: str = "/home/yuqing/Datas/VQA-RAD/test.jsonl"
    vqa_rad_test_jsonl_path: str = "/home/yuqing/Datas/VQA-RAD/test_official.jsonl"
    vqa_rad_image_root: str = "/home/yuqing/Datas/VQA-RAD/images"
    vqa_rad_max_size: int = 1024

    # --- 3. 模型底座与量化 ---
    model_name_or_path: str = "/home/yuqing/Models/Qwen3-VL-8B-Instruct"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"

    # --- 4. 视觉 Adapter 参数 (必须与训练时完全一致) ---
    visual_adapter_hidden_dim: int = 4096
    visual_adapter_r: int = 16
    visual_adapter_alpha: float = 0.1

    # --- 5. 评测与生成参数 (Generation Config) ---
    max_new_tokens: int = 64
    temperature: float = 0.0  # 客观题评测必须用 0.0 贪心解码，禁止随机采样
    do_sample: bool = False
    per_device_eval_batch_size: int = 4
    dataloader_num_workers: int = 4

    def __post_init__(self):
        # 评测结果输出目录也自动实现物理隔离
        base_eval_dir = "/home/yuqing/Models/RouterB_Plus/eval_results_vqa_rad"
        if self.use_visual_adapter:
            self.output_dir = os.path.join(base_eval_dir, "with_visual_adapter")
        else:
            self.output_dir = os.path.join(base_eval_dir, "lora_only_baseline")
        os.makedirs(self.output_dir, exist_ok=True)