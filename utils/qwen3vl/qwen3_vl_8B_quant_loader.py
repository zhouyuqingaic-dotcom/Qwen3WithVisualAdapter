from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration


class Qwen3VLQuantizedLoader:
    """
    Qwen3-VL 量化加载器。

    这个类只负责：
    1. 根据给定配置构造量化参数
    2. 加载 Qwen3-VL 模型
    3. 加载对应的 AutoProcessor
    设计目标：
    ----------
    - 只做“模型加载”这一件事
    - 所有和路径、精度、量化相关的参数都通过 __init__ 传入
    - 保持逻辑清晰、边界明确
    - 方便后续被训练脚本或推理脚本复用
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        processor_path: Optional[Union[str, Path]] = None,
        load_in_4bit: bool = True,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
        bnb_4bit_compute_dtype: Union[str, torch.dtype] = "bfloat16",
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
        attn_implementation: str = "flash_attention_2",
        device_map: Optional[Union[str, dict]] = None,
        trust_remote_code: bool = False,
    ) -> None:
        """
        参数说明
        ----------
        model_path : Union[str, Path]
            Qwen3-VL 模型目录，或 Hugging Face 模型名。
            例如：
            /home/yuqing/Models/Qwen3-VL-8B-Instruct

        processor_path : Optional[Union[str, Path]]
            processor 所在目录。
            如果为 None，则默认与 model_path 相同。

            对于你当前本地模型，一般直接传 None 即可，
            因为 processor 文件通常和模型文件在同一个目录下。

        load_in_4bit : bool
            是否启用 bitsandbytes 的 4-bit 量化加载。
            默认 True。

            注意：
            - True  表示使用 4-bit 量化加载模型
            - False 表示按普通精度加载模型

        bnb_4bit_quant_type : str
            4-bit 量化类型。
            常见值：
            - "nf4"
            - "fp4"

            在 QLoRA / 4-bit 微调场景中，常见选择是 "nf4"。

        bnb_4bit_use_double_quant : bool
            是否启用 double quant（二次量化）。
            默认 True。

            作用：
            - 进一步节省一部分显存
            - 属于 4-bit 量化配置的一部分

        bnb_4bit_compute_dtype : Union[str, torch.dtype]
            4-bit 量化线性层在“计算时”使用的精度。
            常见值：
            - "bfloat16"
            - "float16"
            - "float32"

            注意区分：
            - 权重存储精度：4-bit
            - 前向/反向运算时的计算精度：这里配置的 dtype

        torch_dtype : Optional[Union[str, torch.dtype]]
            传给 from_pretrained 的 torch_dtype。

            如果为 None，则默认等于 bnb_4bit_compute_dtype。
            一般常用：
            - torch.bfloat16
            - torch.float16

            说明：
            这里控制的是模型非量化部分或整体加载时的 dtype 倾向，
            并不意味着“所有参数都严格等于这个 dtype”。

        attn_implementation : str
            attention 实现方式。
            常见值：
            - "flash_attention_2"
            - "sdpa"
            - "eager"

            对你当前已经装好 flash-attn 的环境，
            可以继续使用 "flash_attention_2"。

        device_map : Optional[Union[str, dict]]
            设备映射策略。

            常见用法：
            - "auto"   : 适合快速加载、推理测试
            - None     : 更适合后续训练时由外部代码自行管理设备
            - dict     : 手工指定模块到设备的映射

            建议：
            - 交互式测试 / 单机推理：可用 "auto"

        trust_remote_code : bool
            是否允许信任远程自定义代码。
            对于你当前本地 Qwen3-VL 通常不需要开启，默认 False 即可。
        """
        self.model_path = Path(model_path)
        self.processor_path = Path(processor_path) if processor_path is not None else Path(model_path)

        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        self.bnb_4bit_compute_dtype = self._parse_torch_dtype(bnb_4bit_compute_dtype)

        if torch_dtype is None:
            self.torch_dtype = self.bnb_4bit_compute_dtype
        else:
            self.torch_dtype = self._parse_torch_dtype(torch_dtype)

        self.attn_implementation = attn_implementation
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code

        self._validate_paths()

    def _parse_torch_dtype(self, dtype_value: Union[str, torch.dtype]) -> torch.dtype:
        """
        将字符串形式的 dtype 转成 torch.dtype。

        支持输入：
        ----------
        字符串：
        - "bfloat16"
        - "bf16"
        - "float16"
        - "fp16"
        - "float32"
        - "fp32"

        或直接传入：
        - torch.bfloat16
        - torch.float16
        - torch.float32

        返回：
        ----------
        torch.dtype

        为什么单独写这个函数：
        ----------
        因为 __init__ 中既可能收到字符串，也可能直接收到 torch.dtype。
        单独封装后，主流程更清晰。
        """
        if isinstance(dtype_value, torch.dtype):
            return dtype_value

        if not isinstance(dtype_value, str):
            raise TypeError(
                f"dtype 参数必须是 str 或 torch.dtype，当前收到类型: {type(dtype_value)}"
            )

        value = dtype_value.strip().lower()

        if value in ["bfloat16", "bf16"]:
            return torch.bfloat16
        if value in ["float16", "fp16", "half"]:
            return torch.float16
        if value in ["float32", "fp32"]:
            return torch.float32

        raise ValueError(
            f"不支持的 dtype: {dtype_value}。"
            f"可选值例如: 'bfloat16', 'float16', 'float32'"
        )

    def _validate_paths(self) -> None:
        """
        对模型路径和 processor 路径做最基础的存在性检查。

        这里只检查：
        ----------
        - 路径是否存在

        不检查：
        ----------
        - 模型文件是否完整
        - safetensors 是否损坏
        - config 是否匹配
        - 当前 transformers 版本是否兼容
        - flash-attn 是否已正确安装

        这些问题属于“运行时依赖与环境问题”，
        不属于这里的职责范围。
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"model_path 不存在: {self.model_path}")

        if not self.processor_path.exists():
            raise FileNotFoundError(f"processor_path 不存在: {self.processor_path}")

    def _build_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        构造 4-bit 量化配置。

        返回：
        ----------
        1. 如果 load_in_4bit = True
           返回 BitsAndBytesConfig 对象

        2. 如果 load_in_4bit = False
           返回 None

        逻辑说明：
        ----------
        当使用 bitsandbytes 4-bit 量化时，
        需要把所有量化相关参数集中到一个 BitsAndBytesConfig 中。

        这样做的好处：
        ----------
        - 模型加载逻辑更整洁
        - 量化配置与主加载流程解耦
        - 便于后续单独修改量化参数
        """
        if not self.load_in_4bit:
            return None

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
        )
        return quantization_config

    def load_processor(self) -> AutoProcessor:
        """
        加载 Qwen3-VL 对应的 AutoProcessor。

        返回：
        ----------
        processor : AutoProcessor

        注意：
        ----------
        这里只是把 processor 加载出来，
        不会在这里对图片或文本做任何处理。
        """
        processor = AutoProcessor.from_pretrained(
            self.processor_path,
            trust_remote_code=self.trust_remote_code,
        )
        return processor

    def load_model(self) -> Qwen3VLForConditionalGeneration:
        """
        加载 Qwen3-VL 模型。

        返回：
        ----------
        model : Qwen3VLForConditionalGeneration

        这里做的事情：
        ----------
        1. 根据配置构造 quantization_config
        2. 调用 from_pretrained 加载模型
        3. 返回加载好的 PyTorch 模型对象

        这里不做的事情：
        ----------
        1. 不注入 LoRA
        2. 不设置 model.train() / model.eval()
        3. 不开启 gradient checkpointing
        4. 不包裹 DDP
        5. 不调用 generate

        关于张量与“维度”的说明：
        ----------
        这个阶段只是“加载模型参数”，还没有喂入实际输入，
        因此这里不会产生真正的输入张量，也没有 batch 维度变化。

        你可以把这个阶段理解为：
        ----------
        - 从磁盘读取模型配置与权重
        - 构造一个可执行前向传播的 PyTorch Module

        真正会出现输入张量维度变化的地方，是后续：
        ----------
        - processor(images=..., text=...)
        - model(**inputs)
        - generate(...)

        那些逻辑应该放在外部代码中，而不是这个类里。
        """
        quantization_config = self._build_quantization_config()

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            torch_dtype=self.torch_dtype,
            attn_implementation=self.attn_implementation,
            device_map=self.device_map,
            trust_remote_code=self.trust_remote_code,
        )

        return model

    def load(self) -> Tuple[Qwen3VLForConditionalGeneration, AutoProcessor]:
        """
        一次性加载 model 和 processor。

        返回：
        ----------
        model, processor

        使用方式：
        ----------
        model, processor = loader.load()
        """
        model = self.load_model()
        processor = self.load_processor()
        return model, processor
