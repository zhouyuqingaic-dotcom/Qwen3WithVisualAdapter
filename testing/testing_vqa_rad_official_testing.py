import os
import sys
from collections import Counter


from datas.vqa_rad_datasets import VQARADDataset


def main():
    # 配置刚刚生成的官方测试集路径
    jsonl_path = "/home/yuqing/Datas/VQA-RAD/test_official.jsonl"
    image_root = "/home/yuqing/Datas/VQA-RAD/images"

    print("🚀 [1/3] 启动 VQA-RAD Dataset 官方缝合数据读取测试...")
    print(f"📂 JSONL 路径: {jsonl_path}")

    # 1. 实例化 Dataset
    try:
        dataset = VQARADDataset(jsonl_path=jsonl_path, image_root=image_root)
        print(f"✅ 数据集加载成功！共成功读取 {len(dataset)} 条样本。\n")
    except Exception as e:
        print(f"❌ 数据集加载失败，报错信息: {e}")
        return

    # 2. 全局遍历，统计 Metadata 标签完整性，并校验图片文件是否存在
    print("⏳ [2/3] 正在遍历 Dataset，校验 Metadata 标签完整性与图片物理文件...")
    answer_types = Counter()
    question_types = Counter()
    missing_images = []

    for i in range(len(dataset)):
        sample = dataset[i]  # 调用 __getitem__
        ans_type = sample.get("answer_type", "MISSING")
        q_type = sample.get("question_type", "MISSING")
        image_path = sample.get("image_path", "")

        answer_types[ans_type] += 1
        question_types[q_type] += 1

        # 🛡️ 核心新增：校验图片物理路径是否存在
        if not os.path.exists(image_path):
            missing_images.append(image_path)

    # 打印全局统计信息，核对是否为 CLOSED: 272, OPEN: 179
    print("\n📊 Dataset 读取出的 Answer Type 分布 (核对是否与官方 272/179 完美一致):")
    for k, v in answer_types.items():
        print(f"  👉 {k}: {v} 条")

    print("\n📊 Dataset 读取出的 Question Type 分布 (Top 5):")
    for k, v in question_types.most_common(5):
        print(f"  👉 {k}: {v} 条")

    print("\n🖼️ 图像物理存在性校验结果:")
    if len(missing_images) == 0:
        print(f"  ✅ 完美！所有 {len(dataset)} 张图片文件在本地真实存在！")
    else:
        print(f"  ❌ 警告：发现 {len(missing_images)} 张图片丢失！")
        for img in missing_images[:5]:
            print(f"     - 缺失: {img}")

    # 3. 抽样打印，看看具体的一条数据结构
    print("\n👀 [3/3] 抽样检查：打印前 2 条和最后 1 条的组装字典明细:")

    indices_to_test = [0, 1, len(dataset) - 1]

    for idx in indices_to_test:
        sample = dataset[idx]
        print("=" * 60)
        print(f"🔹 Index       : {sample['index']}")
        print(f"🔹 Image Path  : {sample['image_path']}")
        print(f"🔹 Question    : {sample['question']}")
        print(f"🔹 Answer      : {sample['answer']}")
        print(f"🌟 Question Type: {sample['question_type']} (<- 官方标签)")
        print(f"🌟 Answer Type  : {sample['answer_type']} (<- 核心评测字段)")

    print("=" * 60)
    print("🎉 测试完毕！如果以上输出完美包含了 CLOSED 和 OPEN，且图片 100% 存在，你就可以放心去跑 Eval 啦！")


if __name__ == "__main__":
    main()