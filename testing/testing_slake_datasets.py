"""
testing_slake_datasets.py

用于测试 slake_datasets.py 里的 SLAKEDataset 是否能正确读取 SLAKE 数据集。

这个版本不需要在命令行传入参数。
所有测试配置都直接写在 main() 函数里的 config 字典中。

你的 SLAKE 数据目录结构是：

    /home/yuqing/Datas/SLAKE/Slake1.0
    ├── imgs
    ├── KG
    ├── mask.txt
    ├── train.json
    ├── validate.json
    └── test.json

运行方式：

    python testing_slake_datasets.py

注意：
    testing_slake_datasets.py 需要和 slake_datasets.py 放在同一个目录下，
    或者 slake_datasets.py 所在目录已经加入 PYTHONPATH。
"""

from datas.slake_datasets import SLAKEDataset
import sys
from pathlib import Path


def require_path(path: Path, name: str):
    """
    检查路径是否存在。

    如果数据目录、json 文件或 imgs 目录不存在，直接报错。
    这样可以快速定位是不是路径写错了。
    """

    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def short_text(x, max_len=120):
    """
    截断过长文本，方便在终端里查看。

    有些 question / answer 可能比较长，完整打印会让输出太乱。
    """

    x = "" if x is None else str(x)
    x = x.replace("\n", " ").replace("\r", " ")

    if len(x) > max_len:
        return x[: max_len - 3] + "..."

    return x


def get_sample_field(sample, key, default=None):
    """
    从 sample 中安全地读取字段。

    目前 SLAKEDataset 返回的是 dict。
    这里稍微写得通用一点，如果以后返回对象属性，也能兼容。
    """

    if isinstance(sample, dict):
        return sample.get(key, default)

    return getattr(sample, key, default)


def test_one_split(split_name, json_path, image_root, num_samples, check_all_images):
    """
    测试一个数据划分，例如 train / val / test。

    主要检查：
    1. json 文件是否存在
    2. imgs 目录是否存在
    3. SLAKEDataset 是否可以正常初始化
    4. Dataset 长度是否大于 0
    5. 样本里是否有 question / answer / image_path
    6. image_path 指向的图片是否存在
    """


    print("=" * 80)
    print(f"Testing split: {split_name}")
    print(f"JSON:       {json_path}")
    print(f"Image root: {image_root}")

    # 检查当前 split 的 json 文件是否存在
    require_path(json_path, f"{split_name} json")

    # 检查图片目录是否存在
    require_path(image_root, "image_root")

    # 初始化 Dataset
    # Dataset 层只负责读取 json 和拼 image_path，
    # 不做 tokenizer，也不做 image transform。
    dataset = SLAKEDataset(
        json_path=str(json_path),
        image_root=str(image_root),
    )

    # Dataset 不应该为空
    assert len(dataset) > 0, f"{split_name} dataset is empty"
    print(f"Dataset length: {len(dataset)}")

    # 实际要打印/检查的样本数
    n = min(num_samples, len(dataset))

    # 只统计前 n 个样本中图片缺失的数量
    missing_images_in_printed_samples = 0

    for i in range(n):
        sample = dataset[i]

        # __getitem__ 应该返回 dict
        assert isinstance(sample, dict), (
            f"dataset[{i}] should return a dict, got {type(sample)}"
        )

        # 读取核心字段
        question = get_sample_field(sample, "question")
        answer = get_sample_field(sample, "answer")
        image_path = get_sample_field(sample, "image_path")

        # 检查 question
        assert question is not None and str(question).strip() != "", (
            f"{split_name}[{i}] has empty question"
        )

        # 检查 answer
        assert answer is not None and str(answer).strip() != "", (
            f"{split_name}[{i}] has empty answer"
        )

        # 检查 image_path
        assert image_path is not None and str(image_path).strip() != "", (
            f"{split_name}[{i}] has empty image_path"
        )

        # 检查图片文件是否存在
        image_path = Path(str(image_path))
        image_exists = image_path.exists()

        if not image_exists:
            missing_images_in_printed_samples += 1

        # 打印样本内容，方便人工确认字段是否正确
        print("-" * 80)
        print(f"Sample index: {i}")
        print(f"Question:    {short_text(question)}")
        print(f"Answer:      {short_text(answer)}")
        print(f"Image path:  {image_path}")
        print(f"Image exists: {image_exists}")

        # 可选字段，不强制要求存在
        optional_keys = [
            "question_type",
            "answer_type",
            "image_id",
            "image_name",
        ]

        shown_optional = {
            k: get_sample_field(sample, k)
            for k in optional_keys
            if get_sample_field(sample, k) is not None
        }

        if shown_optional:
            print(f"Optional fields: {shown_optional}")

    if missing_images_in_printed_samples > 0:
        print(
            f"[WARNING] {split_name}: "
            f"{missing_images_in_printed_samples} printed sample image(s) not found."
        )

    # 如果 config 中 check_all_images=True，就检查整个 split 的图片路径
    if check_all_images:
        print("-" * 80)
        print(f"Checking image existence for all {len(dataset)} samples...")

        missing = []

        for i in range(len(dataset)):
            sample = dataset[i]
            image_path = Path(str(get_sample_field(sample, "image_path", "")))

            if not image_path.exists():
                missing.append((i, image_path))

        if missing:
            print(f"[WARNING] Missing images in {split_name}: {len(missing)}")

            # 最多打印前 20 个缺失图片，避免终端输出太长
            for idx, path in missing[:20]:
                print(f"  index={idx}, image_path={path}")

            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more")
        else:
            print(f"All image paths exist for {split_name}.")

    print(f"[OK] {split_name} passed basic checks.")

    return len(dataset)


def main():
    """
    主函数。

    所有原本需要通过命令行传入的信息，
    现在都直接写在 config 字典里。
    """

    config = {
        # SLAKE 数据集根目录
        "slake_root": "/home/yuqing/Datas/SLAKE/Slake1.0",

        # 图片文件夹名
        "image_dir": "imgs",

        # 每个 split 打印和检查几个样本
        "num_samples": 10,

        # 是否检查所有样本的图片路径
        # False：只检查前 num_samples 个样本
        # True：遍历整个 train / val / test 检查所有 image_path
        # "check_all_images": False,
        "check_all_images": True,

        # 三个数据划分对应的 json 文件
        # 注意：SLAKE 的验证集文件名是 validate.json，不是 val.json
        "splits": {
            "train": "train.json",
            "val": "validate.json",
            "test": "test.json",
        },
    }

    slake_root = Path(config["slake_root"]).expanduser().resolve()
    image_root = slake_root / config["image_dir"]

    print("SLAKE root:", slake_root)
    print("Image root:", image_root)
    print("Python executable:", sys.executable)

    # 检查 SLAKE 根目录和 imgs 目录
    require_path(slake_root, "SLAKE root")
    require_path(image_root, "imgs directory")

    counts = {}

    # 依次测试 train / val / test
    for split_name, json_filename in config["splits"].items():
        json_path = slake_root / json_filename

        counts[split_name] = test_one_split(
            split_name=split_name,
            json_path=json_path,
            image_root=image_root,
            num_samples=config["num_samples"],
            check_all_images=config["check_all_images"],
        )

    # 汇总输出每个 split 的样本数量
    print("=" * 80)
    print("All SLAKE splits passed basic tests.")
    print("Split sizes:")

    for split_name, count in counts.items():
        print(f"  {split_name}: {count}")


if __name__ == "__main__":
    main()