import random
import torch
from torch.utils.data import DataLoader, Subset

# 1. 导入配置类 (假设你刚才复制 stage2_train_config_slake.py)
from config.slake.stage2_train_config_slake import Stage2TrainConfig

# 导入我们在 Stage 1 里写好的稳健加载器
from utils.biomedclip.biomed_clip_loader import load_biomedclip

# 2. 导入 SLAKE 数据集
from datas.slake_datasets import SLAKEDataset

# 3. 导入量化加载器 (注意：这里我注释了 Wrapper 的前向传播，因为你可能已经升级到了最新的 MoA Wrapper)
from utils.qwen3vl.qwen3_vl_8B_quant_loader import Qwen3VLQuantizedLoader

# 4. 导入专属 Collator 与答案清洗器
from utils.data_tools.collator.slake.slake_datasets_train_collator import SLAKETrainCollator
from utils.data_tools.prompt_cleaning.slake_answer_cleaning import slake_answer_train_cleaning


def test_slake_cleaning_logic():
    """
    Unit Test: 验证 SLAKE 的清洗器是否按预期工作
    """
    print("🧹 [Unit Test] 正在验证 SLAKE 答案清洗逻辑...")

    test_cases = [
        " MRI ",
        "Yes.",
        "Right  ",
        "Liver, Heart",
        "No"
    ]

    for text in test_cases:
        cleaned = slake_answer_train_cleaning(text)
        print(f"  [原始答案]: '{text}'")
        print(f"  [清洗结果]: '{cleaned}'")

    print("✅ 清洗逻辑 Unit Test 完成！\n" + "=" * 60 + "\n")


def main():
    print("🚀 启动 RouterB_Plus_MoA 架构 SLAKE Train Collator 物理链路测试...\n")

    # =========================================================
    # 0. 优先执行清洗逻辑的 Unit Test
    # =========================================================
    test_slake_cleaning_logic()

    cfg = Stage2TrainConfig()
    # 强制开启 dynamic 模式以测试 BioMed 张量
    cfg.router_mode = "dynamic"

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # =========================================================
    # 1. 加载 BioMedCLIP 工具 (🌟 新增：MoA 架构专属依赖)
    # =========================================================
    print(f"⏳ [1/4] 正在加载 BioMedCLIP 预处理工具 (路径: {cfg.biomedclip_path})...")
    
    # 直接调用现成的加载逻辑，下划线 _ 忽略掉庞大的模型实体，只取我们要的工具
    _, biomed_val_transform, biomed_tokenizer = load_biomedclip(
        biomedclip_path=cfg.biomedclip_path,
        print_rank=cfg.print_rank
    )

    print("✅ BioMedCLIP Tokenizer 与 Transform 加载成功！")

    # =========================================================
    # 2. 加载 SLAKE Dataset
    # =========================================================
    print("\n⏳ [2/4] 正在加载 SLAKE Train Dataset...")
    # 这里假设你测试 train.json
    dataset = SLAKEDataset(
        json_path="/home/yuqing/Datas/SLAKE/Slake1.0/train.json",
        image_root="/home/yuqing/Datas/SLAKE/Slake1.0/imgs"
    )
    print(f"✅ 数据集加载完成，样本总数: {len(dataset)}")

    # =========================================================
    # 3. 加载 Qwen3-VL Processor
    # =========================================================
    print("\n⏳ [3/4] 正在加载 Qwen3-VL Processor...")
    # 测试 Collator 只需要 Processor 即可，跳过加载极其庞大的底座模型，大幅提升调试速度
    loader = Qwen3VLQuantizedLoader(
        model_path=cfg.model_name_or_path,
        processor_path=cfg.model_name_or_path,
        load_in_4bit=cfg.load_in_4bit,
    )
    _, processor = loader.load()  # 假设你的 loader 支持不加载 model
    processor.tokenizer.padding_side = "right"
    print("✅ Processor 加载成功！")

    # =========================================================
    # 4. 初始化并调用 Collator
    # =========================================================
    print("\n⏳ [4/4] 初始化 SLAKETrainCollator 并组装 Batch...")
    collator = SLAKETrainCollator(
        processor=processor,
        cfg=cfg,
        biomed_transform=biomed_val_transform,
        biomed_tokenizer=biomed_tokenizer
    )

    batch_size = 2
    test_indices = random.sample(range(len(dataset)), batch_size)
    subset = Subset(dataset, test_indices)

    dataloader = DataLoader(subset, batch_size=batch_size, collate_fn=collator)
    batch = next(iter(dataloader))

    print("\n🟢 组装好的 Batch Tensor Shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  - {k}: {v.shape} (dtype: {v.dtype})")

    # 特别检查 BioMed 张量
    assert "biomed_image_tensors" in batch, "❌ 缺失 biomed_image_tensors！"
    assert "biomed_text_tokens" in batch, "❌ 缺失 biomed_text_tokens！"
    print("\n🚀 [MoA 专属] BioMedCLIP 张量检查通过！")

    print("\n👀 核心检查: 抽取第一个样本，透视 Labels 掩码机制")
    labels = batch["labels"][0]
    input_ids = batch["input_ids"][0]

    # 把 -100 替换回正常的 token 才能 decode 出来看
    clone_ids = input_ids.clone()
    clone_ids[clone_ids == -100] = processor.tokenizer.pad_token_id or 0
    print("\n  [模型实际看到的完整文本 (包含 User 提问与 Instruction)]:")
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

    print(f"\n🎉 SLAKE 数据与组装物理链路彻底打通！可以发车训练了！")


if __name__ == "__main__":
    main()