from PIL import Image
import torch

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

    def __init__(self, processor, cfg,biomed_transform=None, biomed_tokenizer=None):
        self.processor = processor
        self.cfg = cfg

        # 安全读取相关配置，增加兜底值
        self.max_size = getattr(cfg, "mimic_cxr_max_size", 1024)

        self.router_mode = getattr(cfg, "router_mode", "fixed")
        # 直接接收外部传来的极其轻量的预处理工具，完美兼容多进程
        self.biomed_img_transform = biomed_transform
        self.biomed_tokenizer = biomed_tokenizer

    def __call__(self, batch):
        texts = []
        images = []
        prompt_lengths = []

        # 🎯 新增：用于收集 BioMedCLIP 专属的张量
        biomed_imgs = []
        biomed_txts = []

        for sample in batch:
            # 1. 组装指令文本
            question_text = build_mimic_cxr_prompt(
                instruction=self.cfg.mimic_cxr_instruction_suffix
            )

            # 2. 提取并清洗目标答案
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

            text_prompt = self.processor.apply_chat_template(
                messages_prompt, tokenize=False, add_generation_prompt=True,
            )

            messages_full = messages_prompt + [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer_text}],
                }
            ]

            text_full = self.processor.apply_chat_template(
                messages_full, tokenize=False, add_generation_prompt=False,
            )

            # 4. 读取原始图像
            with Image.open(sample["image_path"]) as pil_img:
                img = pil_img.convert("RGB")

            # =======================================================
            # 🧠 新增：在此处截获原始图片和文本，转化为 BioMed 特征
            # =======================================================
            if self.router_mode == "dynamic":
                # 图像：将其 resize 和 center crop 到 224x224
                biomed_imgs.append(self.biomed_img_transform(img))
                # 文本：Tokenize 用户指令 (取 [0] 去除多余的 batch 维度)
                biomed_txts.append(self.biomed_tokenizer([question_text])[0])

            # 5. Qwen 自身的自适应 resize
            w, h = img.size
            scale = min(self.max_size / max(w, h), 1.0)
            new_w, new_h = int(w * scale), int(h * scale)

            if scale < 1.0:
                img = img.resize((new_w, new_h), Image.BICUBIC)

            # 6. 计算 Qwen prompt 长度
            prompt_inputs = self.processor(
                text=[text_prompt], images=[img], return_tensors="pt", padding=False,
            )
            prompt_lengths.append(prompt_inputs["input_ids"].shape[1])

            texts.append(text_full)
            images.append(img)

        # 7. 构造正式包含全序列的 batch
        batch_inputs = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True,
        )

        # =======================================================
        # 🧠 新增：将 BioMed 特征打包塞入字典，供 Wrapper 劫持提取
        # =======================================================
        if self.router_mode == "dynamic":
            batch_inputs["biomed_image_tensors"] = torch.stack(biomed_imgs)  # shape: [Batch, 3, 224, 224]
            batch_inputs["biomed_text_tokens"] = torch.stack(biomed_txts)    # shape: [Batch, Context_Len]

        # 8. 构造 labels，并精准 mask 掉 prompt 区域
        labels = batch_inputs["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id

        for i in range(len(batch)):
            pad_len = 0
            if self.processor.tokenizer.padding_side == "left":
                pad_len = (batch_inputs["attention_mask"][i] == 0).sum().item()

            mask_end_idx = pad_len + prompt_lengths[i]
            labels[i, :mask_end_idx] = -100

            if pad_token_id is not None:
                labels[i][batch_inputs["input_ids"][i] == pad_token_id] = -100

        batch_inputs["labels"] = labels
        return batch_inputs