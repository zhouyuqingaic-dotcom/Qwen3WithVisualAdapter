import os
from dataclasses import dataclass, field


@dataclass
class Stage2EvalConfig:
    """Stage 2 (VQA-MED-2019) 终极评测配置类"""

    router_mode: str = "fixed"

    # ⚠️ 读取 Stage 2 (VQA-MED-2019) 训练完的最终权重目录
    stage2_weights_dynamic: str = "/home/yuqing/Models/RouterB_Plus_MoA/Stage2_VQA_MED_2019/dynamic/final_weights"
    stage2_weights_fixed: str = "/home/yuqing/Models/RouterB_Plus_MoA/Stage2_VQA_MED_2019/fixed/final_weights"

    # --- 1. 任务协议与 Prompt ---
    vqa_med_2019_instruction_suffix: str = (
        "Answer the question briefly and directly based on the image. Use a short medical term or phrase when possible. "
        "For yes/no questions, answer with yes or no. Do not add unnecessary explanation."
    )

    # --- 2. 数据集配置 (指向带真实答案的 Test 集！) ---
    vqa_med_2019_test_data_path: str = "/home/yuqing/Datas/VQA-Med-2019/test/VQAMed2019Test/VQAMed2019_Test_Questions_w_Ref_Answers.txt"
    vqa_med_2019_test_image_root: str = "/home/yuqing/Datas/VQA-Med-2019/test/VQAMed2019Test/Test_images"
    vqa_med_2019_max_size: int = 1024

    # --- 3. 模型底座与量化 ---
    model_name_or_path: str = "/home/yuqing/Models/Qwen3-VL-8B-Instruct"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"

    biomedclip_path: str = "/home/yuqing/Models/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    biomedclip_model_name: str = "ViT-B-16"

    # --- 4. 视觉 Adapter 参数 ---
    visual_adapter_hidden_dim: int = 4096
    visual_adapter_r: int = 16

    global_adapter_kernel_size: int = 1
    local_adapter_kernel_size: int = 3
    region_adapter_kernel_size: int = 5
    fixed_weights: list[float] = field(default_factory=lambda: [0.33, 0.33, 0.34])

    moe_alpha: float = 0 #0.9 #0.8 #0.7 #0.6 #0.5 #0.4 #0.3 #0.2 #0.1 # 需要测哪个 Alpha 就改哪个

    # --- 5. 评测与生成参数 ---
    max_new_tokens: int = 64
    temperature: float = 0.0  # 贪心解码
    do_sample: bool = False
    per_device_eval_batch_size: int = 4
    dataloader_num_workers: int = 4

    def __post_init__(self):
        # 1. 基础目录定义
        base_eval_dir = "/home/yuqing/Models/RouterB_Plus_MoA/eval_results_vqa_med_2019"
        base_stage2_dynamic = "/home/yuqing/Models/RouterB_Plus_MoA/Stage2_VQA_MED_2019/dynamic"
        base_stage2_fixed = "/home/yuqing/Models/RouterB_Plus_MoA/Stage2_VQA_MED_2019/fixed"

        # 2. 根据 router_mode 提取基础路径
        if self.router_mode == "dynamic":
            base_output_dir = os.path.join(base_eval_dir, "dynamic")
            base_stage2_dir = base_stage2_dynamic
        elif self.router_mode == "fixed":
            base_output_dir = os.path.join(base_eval_dir, "fixed")
            base_stage2_dir = base_stage2_fixed
        else:
            raise ValueError(f"❌ 不支持的 router_mode: {self.router_mode}")

        self.output_dir = f"{base_output_dir}_Alpha_{self.moe_alpha}"
        stage2_alpha_dir = f"{base_stage2_dir}_Alpha_{self.moe_alpha}"
        self.stage2_weights_dir = os.path.join(stage2_alpha_dir, "final_weights")

        print("\n" + "=" * 60)
        print(f"🔬 [Stage 2 Eval Config] VQA-MED-2019 | 模式: {self.router_mode.upper()} | Alpha: {self.moe_alpha}")
        print(f"📂 读取 Stage 2 权重: {self.stage2_weights_dir}")
        print(f"📊 评测结果输出目录: {self.output_dir}")
        print("=" * 60 + "\n")