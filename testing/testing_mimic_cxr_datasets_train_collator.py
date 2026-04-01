import sys
import random
import torch
from torch.utils.data import DataLoader, Subset


# 1. 导入全新的配置类
from config.stage1_train_config import TrainConfig

# 2. 导入数据集
from datas.mimic_cxr_datasets import MIMICCXRDataset

# 3. 导入量化加载器与全新的双臂 Wrapper
from utils.qwen3vl.qwen3_vl_8B_quant_loader import Qwen3VLQuantizedLoader
from utils.qwen3vl.qwen3_vl_8B_lora_wrapper import Qwen3VLLoraAndVisualAdapterWrapper,Qwen3VLLoraWrapper

# 4. 导入专属 Collator 与文本清洗器
from utils.data_tools.collator.mimic_cxr_datasets_train_collator import MIMICCXRTrainCollator
from utils.data_tools.prompt_cleaning.mimic_cxr_text_cleaning import mimic_cxr_text_train_cleaning


def test_mimic_cleaning_logic():
    """
    Unit Test: 验证 MIMIC-CXR 的清洗器是否按预期工作
    """
    print("🧹 [Unit Test] 正在验证 MIMIC-CXR 文本清洗逻辑...")

    test_cases = [
        "IMPRESSION: Mild edema ",
        "[**Name**] no acute disease",
        "none",
        "Prominent hilar vasculature",
        "Findings:   The lungs are clear . ",  # 额外加一条测多余空格和标点前空格的
    ]

    for text in test_cases:
        cleaned = mimic_cxr_text_train_cleaning(text)
        print(f"  [原始文本]: '{text}'")
        print(f"  [清洗结果]: '{cleaned}'\n")

    print("✅ 清洗逻辑 Unit Test 完成！\n" + "=" * 60 + "\n")


def main():
    print("🚀 启动 Route B+ (视觉适配器增强版) MIMIC-CXR Train Collator 物理链路测试...\n")

    # =========================================================
    # 0. 优先执行清洗逻辑的 Unit Test
    # =========================================================
    test_mimic_cleaning_logic()

    cfg = TrainConfig()
    #设置随机种子
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # =========================================================
    # 1. 加载 Dataset (走缓存秒开)
    # =========================================================
    print("⏳ [1/5] 正在加载 MIMIC-CXR Dataset...")
    dataset = MIMICCXRDataset(
        csv_path=cfg.mimic_cxr_metadata_csv,
        image_root=cfg.mimic_cxr_image_root,
        report_root=cfg.mimic_cxr_report_root,
        target_section=cfg.mimic_cxr_target_section,
        load_report=cfg.mimic_cxr_load_report,
        allowed_view_positions=cfg.mimic_cxr_view_positions,
        drop_empty_target=cfg.mimic_cxr_drop_empty_target,
        cache_dir=cfg.mimic_cxr_cache_dir,
        use_indices_cache=cfg.mimic_cxr_use_indices_cache,
        rebuild_indices_cache=cfg.mimic_cxr_rebuild_indices_cache,
        cache_prefix=cfg.mimic_cxr_cache_prefix,
    )
    print(f"✅ 数据集加载完成，样本总数: {len(dataset)}")

    # =========================================================
    # 2. 加载完整模型与 Processor
    # =========================================================
    print("\n⏳ [2/5] 正在加载 Qwen3-VL 4-bit 底座模型与 Processor...")
    loader = Qwen3VLQuantizedLoader(
        model_path=cfg.model_name_or_path,
        processor_path=cfg.model_name_or_path,
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=cfg.bnb_4bit_compute_dtype,
        torch_dtype=cfg.torch_dtype,
        attn_implementation=cfg.attn_implementation,
        device_map="cuda:0",  # 测试时强行锁定单卡
    )
    base_model, processor = loader.load()
    processor.tokenizer.padding_side = "right"  # 训练标准右侧 padding
    print("✅ 模型与 Processor 加载成功！")

    # =========================================================
    # 3. 施加黑魔法：同时注入 LoRA 与 Visual Adapter
    # =========================================================
    print("\n⏳ [3/5] 正在执行双臂注入 (LLM LoRA + Vision Adapter)...")
    if cfg.use_visual_adapter:
        wrapper = Qwen3VLLoraAndVisualAdapterWrapper(
            lora_r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            lora_target_modules=cfg.lora_target_modules,
            gradient_checkpointing=cfg.gradient_checkpointing,
            visual_adapter_hidden_dim=cfg.visual_adapter_hidden_dim,
            visual_adapter_r=cfg.visual_adapter_r,
            visual_adapter_alpha=cfg.visual_adapter_alpha
        )
    else:
        wrapper=Qwen3VLLoraWrapper(
            lora_r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            lora_target_modules=cfg.lora_target_modules,
            gradient_checkpointing=cfg.gradient_checkpointing,
        )



    model = wrapper.wrap(base_model)
    model.eval()  # 我们只测前向通道计算 loss，不更新梯度
    print("✅ 猴子补丁执行完毕，底座模型已升级为 Route B+ 架构！")

    # =========================================================
    # 4. 初始化并调用 Collator
    # =========================================================
    print("\n⏳ [4/5] 初始化 MIMICCXRTrainCollator 并组装 Batch...")
    collator = MIMICCXRTrainCollator(processor, cfg)

    batch_size = 2
    test_indices = random.sample(range(len(dataset)), batch_size)
    subset = Subset(dataset, test_indices)

    dataloader = DataLoader(subset, batch_size=batch_size, collate_fn=collator)
    batch = next(iter(dataloader))

    print("\n🟢 组装好的 Batch Tensor Shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  - {k}: {v.shape} (dtype: {v.dtype})")

    print("\n👀 核心检查: 抽取第一个样本，透视 Labels 掩码机制")
    labels = batch["labels"][0]
    input_ids = batch["input_ids"][0]

    # 把 -100 替换回正常的 token 才能 decode 出来看
    clone_ids = input_ids.clone()
    clone_ids[clone_ids == -100] = processor.tokenizer.pad_token_id or 0
    print("\n  [模型实际看到的完整文本 (包含 User 提问)]:")
    print("  " + "-" * 50)
    print("  " + processor.tokenizer.decode(clone_ids, skip_special_tokens=False))
    print("  " + "-" * 50)

    # 提取真正参与计算 loss 的部分
    valid_label_ids = labels[labels != -100]
    print("\n  [真正参与计算 Loss 的目标文本 (Labels != -100 的部分)]:")
    if len(valid_label_ids) > 0:
        print("  " + "-" * 50)
        print("  " + processor.tokenizer.decode(valid_label_ids, skip_special_tokens=False))
        print("  " + "-" * 50)
    else:
        print("  ⚠️ 严重警告：所有的 label 都是 -100，没有任何文本参与 loss 计算！")

    # =========================================================
    # 5. 模型前向传播 (点火测试)
    # =========================================================
    print("\n🔥 [5/5] 将张量推入 GPU，执行包含 Visual Adapter 的模型前向传播...")
    with torch.no_grad():
        inputs = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss

    print(f"\n🎉 完整物理链路测试通过！")
    print(f"📈 模型成功计算出 Batch Loss: {loss.item():.4f}")
    if torch.isnan(loss):
        print("❌ 警告：Loss 为 NaN，可能存在全 padding 或图像黑图问题！")


if __name__ == "__main__":
    main()