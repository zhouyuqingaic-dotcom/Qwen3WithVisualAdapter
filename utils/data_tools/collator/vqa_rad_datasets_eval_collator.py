import torch
from PIL import Image
from typing import Dict, Any, List, Tuple

# 导入 VQA-RAD 专属的 Prompt 组装器
from utils.data_tools.prompt_builder.vqa_rad_prompt_builder import build_vqa_rad_prompt


class VQARADEvalCollator:
    """
    VQA-RAD 专属的评估/推理 Collator (Eval Collator)。

    与 Train Collator 的区别：
    1. 不再生成 labels (不需要计算 loss)。
    2. 只组装 User 的问题部分 (含图片)，加上 add_generation_prompt=True 让模型准备作答。
    3. 同步返回 metadata_list (包含 Ground Truth 等)，方便评测脚本进行对账。
    """

    def __init__(self, processor, cfg):
        self.processor = processor
        self.cfg = cfg

    def __call__(self, batch: List[Dict[str, Any]]) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
        texts = []
        images = []
        metadata_list = []

        for sample in batch:
            # 1. 组装提问文本 (拼接 Instruction Suffix)
            question_text = build_vqa_rad_prompt(
                question=sample['question'],
                instruction_suffix=self.cfg.vqa_rad_instruction_suffix
            )

            # 2. 构造仅包含 User 提问的消息模板
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": sample["image_path"]},
                        {"type": "text", "text": question_text},
                    ],
                }
            ]

            # 加上生成引导符 (比如 `<|im_start|>assistant\n`)，准备让模型接话
            text_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text_prompt)

            # 3. 图像读取与自适应缩放
            with Image.open(sample["image_path"]) as pil_img:
                img = pil_img.convert("RGB")

            w, h = img.size
            scale = min(self.cfg.vqa_rad_max_size / max(w, h), 1.0)
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.BICUBIC)
            images.append(img)

            # 4. 组装评估必需的 Metadata (把正确答案、原始题目带到外部去打分)
            metadata_list.append({
                "index": sample["index"],
                "image_path": sample["image_path"],
                "question": sample["question"],
                "gt_answer": sample["answer"],
                "question_type": sample.get("question_type", "UNKNOWN"),
                "answer_type": sample.get("answer_type", "UNKNOWN"),  # 👈 增加这一行！
            })

        # 5. 批处理张量化
        batch_inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        return batch_inputs, metadata_list