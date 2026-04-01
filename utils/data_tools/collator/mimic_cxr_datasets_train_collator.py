from PIL import Image

# 导入刚刚新建的专属 Prompt 组装器
from utils.data_tools.prompt_builder.mimic_cxr_prompt_builder import build_mimic_cxr_prompt
# 导入 MIMIC 专属的篇章级文本清洗器
from utils.data_tools.prompt_cleaning.mimic_cxr_text_cleaning import mimic_cxr_text_train_cleaning

class MIMICCXRTrainCollator:
    """
    MIMIC-CXR 专属的 DataCollator (训练专用)。
    处理医学胸片的放射学发现(Findings)或印象(Impression)文本生成。
    完美继承了 PMC-VQA 的安全策略：仅对 Assistant 的最终回答部分计算 Loss，
    彻底屏蔽 User 问题、图像视觉 Token、以及 padding 的梯度。
    """

    def __init__(self, processor, cfg):
        self.processor = processor
        self.cfg = cfg

        # 安全读取相关配置，增加兜底值
        self.max_size = getattr(cfg, "mimic_cxr_max_size", 1024)

    def __call__(self, batch):
        texts = []
        images = []
        prompt_lengths = []

        for sample in batch:
            # 1. 组装指令文本
            question_text = build_mimic_cxr_prompt(
                instruction=self.cfg.mimic_cxr_instruction_suffix
            )

            # 2. 提取目标答案 (MIMIC Dataset 里已经提前抽取好放入 target_text 了)
            # 提取目标答案并进行【专属报告级文本清洗】
            raw_answer = sample.get("target_text", "")
            answer_text = mimic_cxr_text_train_cleaning(raw_answer)

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

            # add_generation_prompt=True 会自动在末尾加上 "<|im_start|>assistant\n" 等前缀
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

            # 4. 读取并自适应 resize 图像
            with Image.open(sample["image_path"]) as pil_img:
                img = pil_img.convert("RGB")

            w, h = img.size
            scale = min(self.cfg.mimic_cxr_max_size / max(w, h), 1.0)
            new_w = int(w * scale)
            new_h = int(h * scale)

            if scale < 1.0:
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

            # 冗余保护：所有 padding 一律不参与 loss 计算
            if pad_token_id is not None:
                labels[i][batch_inputs["input_ids"][i] == pad_token_id] = -100

        batch_inputs["labels"] = labels
        return batch_inputs