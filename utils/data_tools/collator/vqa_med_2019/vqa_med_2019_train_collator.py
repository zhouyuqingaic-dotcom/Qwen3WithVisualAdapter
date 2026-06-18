from PIL import Image
import torch

# 导入 VQA-MED 专属的 Prompt 组装器和清洗器
from utils.data_tools.prompt_builder.vqa_med_2019_prompt_builder import build_vqa_med_2019_prompt
from utils.data_tools.prompt_cleaning.vqa_med_2019_answer_cleaning import vqa_med_2019_answer_train_cleaning


class VQAMED2019TrainCollator:
    """
    VQA-MED-2019 专属的 DataCollator (阶段二训练专用)。
    处理极简的开放式/分类问答，包含轻量级数据清洗。
    仅对 Assistant 的最终回答部分计算 Loss，彻底屏蔽 User 问题、图像视觉 Token 及 padding 的梯度。
    """

    def __init__(self, processor, cfg, biomed_transform=None, biomed_tokenizer=None):
        self.processor = processor
        self.cfg = cfg

        self.router_mode = getattr(cfg, "router_mode", "fixed")
        self.biomed_img_transform = biomed_transform
        self.biomed_tokenizer = biomed_tokenizer

    def __call__(self, batch):
        texts = []
        images = []
        prompt_lengths = []

        biomed_imgs = []
        biomed_txts = []

        for sample in batch:
            # 1. 组装问题与指令
            question_text = build_vqa_med_2019_prompt(
                question=sample['question'],
                instruction_suffix=self.cfg.vqa_med_2019_instruction_suffix
            )

            # 2. 提取目标答案并清洗
            answer_text = vqa_med_2019_answer_train_cleaning(sample["answer"])

            # 3. 构造 prompt-only 与 full-text 两套消息
            messages_prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": sample["image_path"]},
                        {"type": "text", "text": question_text},
                    ],
                }
            ]

            text_prompt = self.processor.apply_chat_template(
                messages_prompt,
                tokenize=False,
                add_generation_prompt=True,
            )

            messages_full = messages_prompt + [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer_text},
                    ],
                }
            ]

            text_full = self.processor.apply_chat_template(
                messages_full,
                tokenize=False,
                add_generation_prompt=False,
            )

            # 4. 读取并 resize 图像
            with Image.open(sample["image_path"]) as pil_img:
                img = pil_img.convert("RGB")

            if self.router_mode == "dynamic":
                biomed_imgs.append(self.biomed_img_transform(img))
                biomed_txts.append(self.biomed_tokenizer([question_text])[0])

            w, h = img.size
            scale = min(self.cfg.vqa_med_2019_max_size / max(w, h), 1.0)
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.BICUBIC)

            # 5. 计算 prompt 长度
            prompt_inputs = self.processor(
                text=[text_prompt],
                images=[img],
                return_tensors="pt",
                padding=False,
            )
            prompt_lengths.append(prompt_inputs["input_ids"].shape[1])

            texts.append(text_full)
            images.append(img)

        # 6. 构造包含全序列的 batch
        batch_inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        if self.router_mode == "dynamic":
            batch_inputs["biomed_image_tensors"] = torch.stack(biomed_imgs)
            batch_inputs["biomed_text_tokens"] = torch.stack(biomed_txts)

        # 7. 构造 labels，并精准 mask 掉 prompt 区域
        labels = batch_inputs["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id

        for i in range(len(batch)):
            # 🚀 极致优化：直接计算左侧 padding 的长度，不做废话判断
            pad_len = (batch_inputs["attention_mask"][i] == 0).sum().item()
            mask_end_idx = pad_len + prompt_lengths[i]

            # Mask 掉 System 提示词、图片特征、User 提问
            labels[i, :mask_end_idx] = -100

            # 兜底 Mask 掉所有的 Padding Token
            if pad_token_id is not None:
                labels[i][batch_inputs["input_ids"][i] == pad_token_id] = -100

        batch_inputs["labels"] = labels
        return batch_inputs