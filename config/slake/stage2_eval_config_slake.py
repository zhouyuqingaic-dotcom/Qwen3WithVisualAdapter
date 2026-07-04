import os
from dataclasses import dataclass, field


@dataclass
class Stage2EvalConfig:
    """Stage 2 (SLAKE) 终极评测配置类 (双模式切换)"""

    # =========================================================
    # ✨ 核心创新点开关：选择当前评测的模型模式
    # =========================================================
    # 可选: "dynamic" (动态MoE) 或 "fixed" (静态MoE)
    router_mode: str = "fixed" #"dynamic" #"dynamic"

    # ⚠️ 读取 Stage 2 (SLAKE) 训练完的最终权重目录
    # stage2_weights_dynamic: str = "/home/yuqing/Models/RouterB_Plus_MoA/Stage2_SLAKE/dynamic/final_weights"
    # stage2_weights_fixed: str = "/home/yuqing/Models/RouterB_Plus_MoA/Stage2_SLAKE/fixed/final_weights"

    stage2_weights_dynamic: str = "/home/yuqing/Models/RouterB_Plus_MoA/Stage2_SLAKE/dynamic_Alpha_1_seed_1912/final_weights"
    stage2_weights_fixed: str = "/home/yuqing/Models/RouterB_Plus_MoA/Stage2_SLAKE/fixed_Alpha_0_seed_1912/final_weights"

    # --- 1. 任务协议与 Prompt (与训练期无缝对齐) ---
    slake_instruction_suffix: str = (
        "Answer the question briefly and directly based on the image. Use a short medical term or phrase when possible. "
        "For yes/no questions, answer with yes or no. Do not add unnecessary explanation."
    )

    # --- 2. 数据集配置 (指向 SLAKE 测试集) ---
    slake_test_json_path: str = "/home/yuqing/Datas/SLAKE/Slake1.0/test.json"
    slake_val_json_path: str = "/home/yuqing/Datas/SLAKE/Slake1.0/validate.json"
    slake_image_root: str = "/home/yuqing/Datas/SLAKE/Slake1.0/imgs"
    slake_max_size: int = 1024

    # --- 3. 模型底座与量化 ---
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

    # --- 4. 视觉 Adapter 参数 (必须与训练时完全一致) ---
    visual_adapter_hidden_dim: int = 4096
    visual_adapter_r: int = 16

    global_adapter_kernel_size: int = 1
    local_adapter_kernel_size: int = 3
    region_adapter_kernel_size: int = 5
    # 🚀 与 Train Config 绝对对齐的硬融合比例
    fixed_weights: list[float] = field(default_factory=lambda: [0.333, 0.333, 0.334])

    # 🚀 【核心锁死】需要评测哪组 Alpha 就填哪组，这里默认填
    moe_alpha: float = 0 #0.3 #0.2 #0.3 #0.4 #0.5 #0.6 #0.7 #0.8 #0.9 #1 #0 #1 #0.1 #0 #0.2 #0.7 #0.3 #0.4 #0.5 #0.6 #0.8 #0.9 #1

    # --- 5. 评测与生成参数 (Generation Config) ---
    max_new_tokens: int = 64
    temperature: float = 0.0  # 客观题评测必须用 0.0 贪心解码，禁止随机采样
    do_sample: bool = False
    per_device_eval_batch_size: int = 4
    dataloader_num_workers: int = 4


    # def __post_init__(self):
    #     # 1. 基础目录定义 (Base Directories - 替换为 SLAKE 专属)
    #     base_eval_dir = "/home/yuqing/Models/RouterB_Plus_MoA/eval_results_slake"
    #     base_stage2_dynamic = "/home/yuqing/Models/RouterB_Plus_MoA/Stage2_SLAKE/dynamic"
    #     base_stage2_fixed = "/home/yuqing/Models/RouterB_Plus_MoA/Stage2_SLAKE/fixed"
    #
    #     # 🚀 2. 根据 router_mode 提取基础路径
    #     if self.router_mode == "dynamic":
    #         base_output_dir = os.path.join(base_eval_dir, "dynamic")
    #         base_stage2_dir = base_stage2_dynamic
    #     elif self.router_mode == "fixed":
    #         base_output_dir = os.path.join(base_eval_dir, "fixed")
    #         base_stage2_dir = base_stage2_fixed
    #     else:
    #         raise ValueError(f"❌ 不支持的 router_mode: {self.router_mode}，只能是 'dynamic' 或 'fixed'")
    #
    #     # 🚀 3. 给评测结果的【保存路径】追加 Alpha 后缀 (例如: eval_results_slake/dynamic_Alpha_0.9)
    #     self.output_dir = f"{base_output_dir}_Alpha_{self.moe_alpha}"
    #
    #     # 🚀 4. 给 Stage 2 权重的【读取路径】追加 Alpha 后缀，并在最末端拼接 "final_weights"
    #     # 你的寻宝脚本将会去这个目录里面扫荡 checkpoint
    #     stage2_alpha_dir = f"{base_stage2_dir}_Alpha_{self.moe_alpha}"
    #     self.stage2_weights_dir = os.path.join(stage2_alpha_dir, "final_weights")
    #
    #     # =========================================================
    #     # 🖨️ 新增：打印最终生成的路径，方便终端核对
    #     # =========================================================
    #     print("\n" + "=" * 60)
    #     print(f"🔬 [Stage 2 Eval Config] 初始化完成 | 模式: {self.router_mode.upper()} | Alpha: {self.moe_alpha}")
    #     print(f"📂 读取 Stage 2 权重: {self.stage2_weights_dir}")
    #     print(f"📊 评测结果输出目录: {self.output_dir}")
    #     print("=" * 60 + "\n")

    # 🚀 当前要评测的 seed
    seed: int = 1912 #1024 #2048 #1024

    def __post_init__(self):
        base_eval_dir = "/home/yuqing/Models/RouterB_Plus_MoA/eval_results_slake"
        base_stage2_dir = "/home/yuqing/Models/RouterB_Plus_MoA/Stage2_SLAKE"

        alpha_str = f"{self.moe_alpha:g}"

        if self.router_mode == "dynamic":
            run_name = f"dynamic_Alpha_{alpha_str}_seed_{self.seed}"
        elif self.router_mode == "fixed":
            run_name = f"fixed_Alpha_{alpha_str}_seed_{self.seed}"
        else:
            raise ValueError(f"❌ 不支持的 router_mode: {self.router_mode}")

        self.output_dir = os.path.join(base_eval_dir, run_name)
        self.stage2_weights_dir = os.path.join(base_stage2_dir, run_name, "final_weights")

        print("\n" + "=" * 60)
        print(
            f"🔬 [Stage 2 Eval Config] 初始化完成 | 模式: {self.router_mode.upper()} | Alpha: {self.moe_alpha} | Seed: {self.seed}")
        print(f"📂 读取 Stage 2 权重: {self.stage2_weights_dir}")
        print(f"📊 评测结果输出目录: {self.output_dir}")
        print("=" * 60 + "\n")