import os
import json
import open_clip
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
from transformers import AutoTokenizer

from utils.qwen3vl.qwen3_vl_8B_lora_wrapper import FrozenBioMedCLIPFeatureExtractor
from utils.ddp.ddp_utils import ddp_print


def load_biomedclip(biomedclip_path: str, print_rank: int = 0):
    """
    独立加载 BioMedCLIP 的全局组件。

    Args:
        biomedclip_path (str): BioMedCLIP 本地权重的绝对路径。
        print_rank (int): 用于 DDP 打印的进程等级。

    Returns:
        tuple: (biomed_extractor, biomed_transform, biomed_tokenizer)
    """
    ddp_print(f"\n⏳ [Loader] 正在通过本地注册表加载 BioMedCLIP (路径: {biomedclip_path})",
              print_rank=print_rank)

    try:
        # 1. 定位配置文件与权重文件
        cfg_path = os.path.join(biomedclip_path, "open_clip_config.json")
        bin_path = os.path.join(biomedclip_path, "open_clip_pytorch_model.bin")

        with open(cfg_path, "r", encoding="utf-8") as f:
            clip_cfg = json.load(f)

        model_cfg = clip_cfg["model_cfg"]
        preprocess_cfg = clip_cfg["preprocess_cfg"]
        custom_model_name = "biomedclip_local"

        # 2. 【核心绝杀】：将配置注入到 open_clip 全局注册表
        if (not custom_model_name.startswith(HF_HUB_PREFIX)
                and custom_model_name not in _MODEL_CONFIGS):
            _MODEL_CONFIGS[custom_model_name] = model_cfg

        # 3. 使用注册的名称和解析出的图像预处理参数加载模型
        raw_biomed_model, _, biomed_transform = open_clip.create_model_and_transforms(
            model_name=custom_model_name,
            pretrained=bin_path,
            **{f"image_{k}": v for k, v in preprocess_cfg.items()},
        )

        # 4. 弃用 CLIP Tokenizer，加载配套的 PubMedBERT Tokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained(biomedclip_path)

        def custom_biomed_tokenizer(texts):
            return hf_tokenizer(
                texts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt"
            )["input_ids"]

        # 5. 包装成提取器
        biomed_extractor = FrozenBioMedCLIPFeatureExtractor(raw_biomed_model)
        ddp_print("✅ BioMedCLIP 全局组件与 PubMedBERT Tokenizer 加载成功！", print_rank=print_rank)

        return biomed_extractor, biomed_transform, custom_biomed_tokenizer

    except Exception as e:
        ddp_print(f"❌ BioMedCLIP 加载失败: {e}", print_rank=print_rank)
        raise e