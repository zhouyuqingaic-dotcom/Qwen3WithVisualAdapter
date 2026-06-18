
import os
from datas.vqa_med_2019_datasets import VQAMED2019Dataset


def test_split(split_name: str, data_dir: str, image_dir: str):
    print("\n" + "🚀" * 30)
    print(f"🔍 正在校验切片: 【{split_name.upper()}】")
    print(f"   📂 Data Path : {data_dir}")
    print(f"   🖼️ Image Dir : {image_dir}")
    print("=" * 60)

    try:
        dataset = VQAMED2019Dataset(data_path=data_dir, image_root=image_dir)
        print(f"✅ 数据集加载成功！共找到 {len(dataset)} 条问答对。")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print("💡 请检查该切片的路径是否正确！")
        return

    # 📊 统计一下题目类型和 Closed/Open 比例
    stats = {"Modality": 0, "Plane": 0, "Organ": 0, "Abnormality": 0, "unknown": 0}
    closed_count = 0

    for i in range(len(dataset)):
        sample = dataset[i]
        # 使用 get 兜底，防止出现意料之外的 Category
        c_type = sample.get("question_type", "unknown")
        if c_type not in stats:
            stats[c_type] = 0
        stats[c_type] += 1

        if sample["answer_type"] == "CLOSED":
            closed_count += 1

    print("\n📊 数据集分类统计:")
    for k, v in stats.items():
        if v > 0:
            print(f"  - {k}: {v} 条")
    print(f"\n🔘 Closed (Yes/No) 题型数量: {closed_count}")
    print(f"📖 Open 题型数量: {len(dataset) - closed_count}")
    print("-" * 40)

    # 🧪 验证图片文件是否全部物理存在 (静默模式，只在报错时大叫)
    print("🔬 正在进行全量图片物理存在校验...")
    missing_count = 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if not os.path.exists(sample['image_path']):
            missing_count += 1
            if missing_count <= 5:  # 只打印前 5 个找不到的，防止刷屏
                print(f"  🔴 [Sample {idx}] 找不到图片: {sample['image_path']}")

    if missing_count == 0:
        print(f"  🟢 全量图片物理校验: 100% SUCCESS! (共 {len(dataset)} 张图片真实存在)")
    else:
        raise Exception(f"  ❌ 致命错误: 共有 {missing_count} 张图片在硬盘上找不到！")

    # 打印一条头、一条尾的抽样看看格式
    print("\n🔬 抽样检查首尾数据格式:")
    for idx in [0, len(dataset) - 1]:
        sample = dataset[idx]
        print(f"  [Sample {idx}]")
        print(f"  ❓ Q: {sample['question']}")
        print(f"  💡 A: {sample['answer']}")
        print(f"  🏷️ Q_Type: {sample['question_type']} | A_Type: {sample['answer_type']}\n")


def main():
    # =========================================================================
    # 🎯 设定三个切片的路径 (请根据你 ls 看到的实际解压结构微调这里的路径)
    # =========================================================================

    base_dir = "/home/yuqing/Datas/VQA-Med-2019"

    splits_config = {
        "Train": {
            "data": f"{base_dir}/train/ImageClef-2019-VQA-Med-Training/QAPairsByCategory",
            "img": f"{base_dir}/train/ImageClef-2019-VQA-Med-Training/Train_images"
        },
        "Validation": {
            "data": f"{base_dir}/val/ImageClef-2019-VQA-Med-Validation/QAPairsByCategory",
            "img": f"{base_dir}/val/ImageClef-2019-VQA-Med-Validation/Val_images"
        },
        "Test": {
            # 🎯 改动在这里！直接指向官方带标准答案和分类的豪华版 TXT
            "data": f"{base_dir}/test/VQAMed2019Test/VQAMed2019_Test_Questions_w_Ref_Answers.txt",
            "img": f"{base_dir}/test/VQAMed2019Test/Test_images"
        }
    }

    # 批量跑校验
    for split_name, paths in splits_config.items():
        # 如果路径存在才去测，防止因为你解压的文件夹名字不对而直接崩溃
        if os.path.exists(paths["data"]) and os.path.exists(paths["img"]):
            test_split(split_name, paths["data"], paths["img"])
        else:
            print("\n" + "⚠️" * 15)
            print(f"⏩ 跳过 {split_name} 切片: 路径在硬盘上不存在，请检查脚本里的路径配置！")
            print(f"   预期 Data: {paths['data']}")
            print(f"   预期 Img : {paths['img']}")
            print("⚠️" * 15)


if __name__ == "__main__":
    main()