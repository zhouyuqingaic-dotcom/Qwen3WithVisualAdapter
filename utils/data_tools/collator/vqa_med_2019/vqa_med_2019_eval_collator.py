import torch
from PIL import Image
from typing import Dict, Any, List, Tuple

from utils.data_tools.prompt_builder.vqa_med_2019_prompt_builder import build_vqa_med_2019_prompt


class VQAMED2019EvalCollator:
    """
    VQA-MED-2019 专属的评估/推理 Collator (Eval Collator)。

    修复点：
    1. 保留 28 像素对齐逻辑，降低 Qwen3-VL RoPE / grid 越界风险。
    2. 使用 processor 的 max_pixels 继续约束视觉 token 数。
    3. DYNAMIC 模式下正确注入 BioMedCLIP 图像与文本张量。
    4. 将 padding=False 改为 padding=True，避免 batch_size > 1 时
       text / image token / image_grid_thw 对齐不稳定。
    5. 增加输入合法性检查，提前暴露 BioMedCLIP transform/tokenizer 缺失问题。
    """

    def __init__(self, processor, cfg, biomed_transform=None, biomed_tokenizer=None):
        self.processor = processor
        self.cfg = cfg

        self.router_mode = getattr(cfg, "router_mode", "fixed")
        self.biomed_img_transform = biomed_transform
        self.biomed_tokenizer = biomed_tokenizer

        if self.router_mode == "dynamic":
            if self.biomed_img_transform is None:
                raise ValueError("router_mode='dynamic' 时必须传入 biomed_transform")
            if self.biomed_tokenizer is None:
                raise ValueError("router_mode='dynamic' 时必须传入 biomed_tokenizer")

    def _resize_to_qwen_grid(self, img: Image.Image) -> Image.Image:
        """
        将图像按最长边缩放，并强制宽高对齐到 28 的倍数。

        Qwen-VL 系列视觉分支通常依赖 patch / merge 后的 grid。
        宽高不稳定或过大时，容易在 RoPE / gather 阶段触发 CUDA index out of bounds。
        """
        w, h = img.size
        max_size = int(getattr(self.cfg, "vqa_med_2019_max_size", 672))

        scale = min(max_size / max(w, h), 1.0)
        new_w, new_h = int(w * scale), int(h * scale)

        # 强制对齐到 28 的倍数；至少保留 28，避免极端小图变成 0。
        new_w = max(28, (new_w // 28) * 28)
        new_h = max(28, (new_h // 28) * 28)

        # 如果原图本身已经符合尺寸，也仍然 resize 到对齐后的尺寸，保证 processor 输入稳定。
        return img.resize((new_w, new_h), Image.BICUBIC)

    def __call__(self, batch: List[Dict[str, Any]]) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
        texts: List[str] = []
        images: List[Image.Image] = []
        metadata_list: List[Dict[str, Any]] = []

        biomed_imgs: List[torch.Tensor] = []
        biomed_txts: List[torch.Tensor] = []

        for sample in batch:
            # 1. 组装提问文本
            question_text = build_vqa_med_2019_prompt(
                question=sample["question"],
                instruction_suffix=self.cfg.vqa_med_2019_instruction_suffix,
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

            text_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text_prompt)

            # 3. 图像读取、RGB 转换、物理缩放、28 像素对齐
            with Image.open(sample["image_path"]) as pil_img:
                img = pil_img.convert("RGB")

            img = self._resize_to_qwen_grid(img)
            images.append(img)

            # 4. DYNAMIC 路由模式下，为 Stage 2 MoE 注入 BioMedCLIP 输入
            if self.router_mode == "dynamic":
                biomed_img = self.biomed_img_transform(img)
                biomed_txt = self.biomed_tokenizer([question_text])[0]

                if not isinstance(biomed_img, torch.Tensor):
                    raise TypeError(f"biomed_img_transform 必须返回 torch.Tensor，实际得到: {type(biomed_img)}")
                if not isinstance(biomed_txt, torch.Tensor):
                    raise TypeError(f"biomed_tokenizer 必须返回 torch.Tensor，实际得到: {type(biomed_txt)}")

                biomed_imgs.append(biomed_img)
                biomed_txts.append(biomed_txt)

            # 5. 组装评估必需的 metadata
            metadata_list.append(
                {
                    "index": sample["index"],
                    "image_path": sample["image_path"],
                    "question": sample["question"],
                    "gt_answer": sample["answer"],
                    "question_type": sample.get("question_type", "UNKNOWN"),
                    "answer_type": sample.get("answer_type", "UNKNOWN"),
                }
            )

        # 6. 批处理张量化
        #
        # 关键修复：
        # 原来 padding=False 在 batch_size > 1 时容易让 Qwen3-VL 的 image token、
        # image_grid_thw、pixel_values 对齐不稳定。
        #
        # 这里改成 padding=True。即使 batch_size=1，也不会伤害结果；
        # 如果之后 batch_size 改回 4，也更安全。
        max_size = int(getattr(self.cfg, "vqa_med_2019_max_size", 672))
        batch_inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            max_pixels=int(max_size ** 2),
        )

        # 7. 注入 BioMedCLIP 特征张量
        if self.router_mode == "dynamic":
            batch_inputs["biomed_image_tensors"] = torch.stack(biomed_imgs, dim=0)
            batch_inputs["biomed_text_tokens"] = torch.stack(biomed_txts, dim=0)

        return batch_inputs, metadata_list