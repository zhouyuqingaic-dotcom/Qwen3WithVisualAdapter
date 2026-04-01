from PIL import Image

# 导入我们刚刚新建的 VQA-RAD 专属的 Prompt 组装器
from utils.data_tools.prompt_builder.vqa_rad_prompt_builder import build_vqa_rad_prompt
# 导入我们刚刚新建的 VQA-RAD 专属极轻量答案清洗器
from utils.data_tools.prompt_cleaning.vqa_rad_answer_cleaning import vqa_rad_answer_train_cleaning


class VQARADTrainCollator:
    """
    VQA-RAD 专属的 DataCollator (阶段二训练专用)。
    处理极简的开放式问答，包含针对 VQA-RAD 的轻量级数据清洗。
    仅对 Assistant 的最终回答部分计算 Loss，彻底屏蔽 User 问题、图像视觉 Token 及 padding 的梯度。
    """

    def __init__(self, processor, cfg):
        self.processor = processor
        self.cfg = cfg

    def __call__(self, batch):
        texts = []
        images = []
        prompt_lengths = []

        for sample in batch:
            # 1. 组装问题与指令 (调用 VQA-RAD 专属极简 Builder)
            question_text = build_vqa_rad_prompt(
                question=sample['question'],
                instruction_suffix=self.cfg.vqa_rad_instruction_suffix
            )

            # 2. 提取目标答案，并套上 VQA-RAD 极轻量清洗逻辑
            # 这里能完美解决比如 "skull \tcartilage" 这种携带制表符的脏数据
            answer_text = vqa_rad_answer_train_cleaning(sample["answer"])

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

            w, h = img.size
            scale = min(self.cfg.vqa_rad_max_size / max(w, h), 1.0)
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.BICUBIC)

            # 5. 用同源 processor 精确计算 prompt 长度
            # padding=False，确保算出来的是绝对准确的单条样本长度
            prompt_inputs = self.processor(
                text=[text_prompt],
                images=[img],
                return_tensors="pt",
                padding=False,
            )
            prompt_lengths.append(prompt_inputs["input_ids"].shape[1])

            texts.append(text_full)
            images.append(img)

        # 6. 构造正式包含全序列的 batch
        batch_inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        # 7. 构造 labels，并精准 mask 掉 prompt 区域
        labels = batch_inputs["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id

        for i in range(len(batch)):
            pad_len = 0
            # 使用 attention_mask 判断 padding，比直接查 pad_token_id 更鲁棒
            if self.processor.tokenizer.padding_side == "left":
                pad_len = (batch_inputs["attention_mask"][i] == 0).sum().item()

            mask_end_idx = pad_len + prompt_lengths[i]

            # mask 掉 prompt（包含图片 token、user 文本、assistant 模板前缀）
            labels[i, :mask_end_idx] = -100

            # 冗余保护：屏蔽所有 padding 不参与 loss 计算
            labels[i][batch_inputs["input_ids"][i] == pad_token_id] = -100

        batch_inputs["labels"] = labels
        return batch_inputs