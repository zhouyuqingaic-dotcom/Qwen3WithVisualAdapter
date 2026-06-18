
# 导入你的 Dataset 和 Collators
from datas.vqa_med_2019_datasets import VQAMED2019Dataset
from utils.data_tools.collator.vqa_med_2019.vqa_med_2019_train_collator import VQAMED2019TrainCollator
from utils.data_tools.collator.vqa_med_2019.vqa_med_2019_eval_collator import VQAMED2019EvalCollator

# 导入底座和 BioMedCLIP 加载器
from utils.qwen3vl.qwen3_vl_8B_quant_loader import Qwen3VLQuantizedLoader
from utils.biomedclip.biomed_clip_loader import load_biomedclip


class MockConfig:
    """模拟一个精简版的 Config，专供测试 Collator 使用"""
    model_name_or_path = "/home/yuqing/Models/Qwen3-VL-8B-Instruct"
    biomedclip_path = "/home/yuqing/Models/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

    # 模拟 VQA-MED 配置
    vqa_med_2019_instruction_suffix = (
        "Answer the question briefly and directly based on the image. "
        "Use a short medical term or phrase when possible. "
        "For yes/no questions, answer with yes or no."
    )
    vqa_med_2019_max_size = 1024

    # 强制开启 dynamic 测试最复杂的特征提取逻辑
    router_mode = "dynamic"

    # Qwen 量化参数 (仅加载 processor 用，可随便填)
    load_in_4bit = True
    bnb_4bit_quant_type = "nf4"
    bnb_4bit_use_double_quant = True
    bnb_4bit_compute_dtype = "bfloat16"
    torch_dtype = "bfloat16"
    attn_implementation = "flash_attention_2"


def main():
    print("\n" + "=" * 60)
    print("🚀 启动 VQA-MED-2019 Collators 终极测试")
    print("=" * 60)

    cfg = MockConfig()

    # 1. 加载 Processor 和 Tokenizer (不需要加载庞大的 LLM 模型)
    print("⏳ 正在加载 Qwen3-VL Processor...")
    loader = Qwen3VLQuantizedLoader(
        model_path=cfg.model_name_or_path, processor_path=cfg.model_name_or_path,
        load_in_4bit=cfg.load_in_4bit, bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant, bnb_4bit_compute_dtype=cfg.bnb_4bit_compute_dtype,
        torch_dtype=cfg.torch_dtype, attn_implementation=cfg.attn_implementation, device_map="cpu"
    )
    processor = loader.load_processor()
    processor.tokenizer.padding_side = "left"

    print("⏳ 正在加载 BioMedCLIP 预处理工具...")
    _, biomed_transform, biomed_tokenizer = load_biomedclip(cfg.biomedclip_path)

    # 2. 准备一小撮 Dataset (我们用 Train 切片做测试)
    data_dir = "/home/yuqing/Datas/VQA-Med-2019/train/ImageClef-2019-VQA-Med-Training/QAPairsByCategory"
    image_dir = "/home/yuqing/Datas/VQA-Med-2019/train/ImageClef-2019-VQA-Med-Training/Train_images"
    dataset = VQAMED2019Dataset(data_path=data_dir, image_root=image_dir)

    # 提取两笔数据组成一个 Batch
    test_batch = []
    for i in range(20):
        test_batch.append(dataset[i*100])

    print("\n" + "🟢" * 20 + " [测试 1: Train Collator] " + "🟢" * 20)
    train_collator = VQAMED2019TrainCollator(processor, cfg, biomed_transform, biomed_tokenizer)
    train_inputs = train_collator(test_batch)

    print("✅ Train Batch 组装完成，正在检查 Tensor 形状：")
    print(f"  - input_ids:            {train_inputs['input_ids'].shape}")
    print(f"  - attention_mask:       {train_inputs['attention_mask'].shape}")
    print(f"  - labels:               {train_inputs['labels'].shape}")
    if cfg.router_mode == "dynamic":
        print(f"  - biomed_image_tensors: {train_inputs['biomed_image_tensors'].shape}")
        print(f"  - biomed_text_tokens:   {train_inputs['biomed_text_tokens'].shape}")

    # 验证 Labels Mask 逻辑
    print("\n🔬 [硬核探伤] 验证 Labels 屏蔽 (Mask) 逻辑 (-100):")
    for i in range(len(test_batch)):
        labels = train_inputs['labels'][i]
        valid_labels = labels[labels != -100]
        decoded_answer = processor.tokenizer.decode(valid_labels, skip_special_tokens=True)
        print(f"  样本 {i} 真正参与计算 Loss 的有效文本 -> [{decoded_answer}]")

    print("\n" + "🔵" * 20 + " [测试 2: Eval Collator] " + "🔵" * 20)
    eval_collator = VQAMED2019EvalCollator(processor, cfg, biomed_transform, biomed_tokenizer)
    eval_inputs, metadata_list = eval_collator(test_batch)

    print("✅ Eval Batch 组装完成，正在检查 Tensor 形状：")
    print(f"  - input_ids:            {eval_inputs['input_ids'].shape}")
    print(f"  - attention_mask:       {eval_inputs['attention_mask'].shape}")
    if cfg.router_mode == "dynamic":
        print(f"  - biomed_image_tensors: {eval_inputs['biomed_image_tensors'].shape}")
        print(f"  - biomed_text_tokens:   {eval_inputs['biomed_text_tokens'].shape}")

    print("\n🔬 [硬核探伤] 验证 Eval Metadata 流转与生成引导符:")
    for i in range(len(test_batch)):
        meta = metadata_list[i]
        input_ids = eval_inputs['input_ids'][i]
        # 去掉 padding 看最后一部分
        valid_input_ids = input_ids[eval_inputs['attention_mask'][i] == 1]
        tail_text = processor.tokenizer.decode(valid_input_ids[-10:])

        print(f"  [样本 {i}] Metadata 透传: ")
        print(f"    - Category : {meta['question_type']}")
        print(f"    - A_Type   : {meta['answer_type']}")
        print(f"    - GT_Answer: {meta['gt_answer']}")
        print(f"    - Prompt 尾部探测 (需看到 assistant 引导符): {repr(tail_text)}")


if __name__ == "__main__":
    main()