import types
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import PreTrainedModel

# 导入自定义视觉残差适配器
from utils.qwen3vl.qwen3_vl_8B_visual_adapter import Qwen3VLVisualAdapter

class Qwen3VLLoraWrapper:
    """
    Qwen3-VL 的 PEFT (LoRA) 包装器。
    负责在 4-bit 量化模型上挂载 LoRA 适配器，并正确配置梯度检查点。
    """

    def __init__(self, lora_r, lora_alpha, lora_dropout, lora_target_modules, gradient_checkpointing):
        self.r = lora_r
        self.alpha = lora_alpha
        self.dropout = lora_dropout
        self.target_modules = lora_target_modules
        self.gradient_checkpointing = gradient_checkpointing

    def wrap(self, model: PreTrainedModel) -> PreTrainedModel:
        # 1. 配置 k-bit 训练环境与梯度检查点
        # 显式传递 use_reentrant=False 以防 DDP 环境下的梯度重复标记错误
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=self.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        # 开启 gradient checkpointing 时须关闭 KV Cache
        if hasattr(model, "config"):
            model.config.use_cache = False

        # 2. 构造并注入 LoRA
        lora_config = LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            target_modules=self.target_modules,
            lora_dropout=self.dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        peft_model = get_peft_model(model, lora_config)

        # 3. 确保视觉编码器等冻结组件的输入层支持梯度反传
        if hasattr(peft_model, "enable_input_require_grads"):
            peft_model.enable_input_require_grads()

        peft_model.print_trainable_parameters()
        return peft_model



class Qwen3VLLoraAndVisualAdapterWrapper:
    """
    Qwen3-VL 专属的 Dual-Tuning (双塔微调) 包装器。

    核心职责：
    1. 在语言大模型 (LLM) 端，标准挂载 LoRA 适配器。
    2. 在视觉编码器 (Vision Tower) 端，通过“猴子补丁 (Monkey Patching)”动态植入轻量级的视觉残差适配器 (Visual Adapter)。

    设计准则 (完美兼容 DDP 分布式训练)：
    - 【法则1：上户口】将 Adapter 作为 `vision_tower` 的属性挂载，使其被纳入 PyTorch 的参数树 (model.parameters())，从而被 DDP 识别并分配同步桶。
    - 【法则2：早买票】在模型交由 HF Trainer 构建 Optimizer 之前完成植入，让 Trainer 自动接管 Adapter 的参数更新。
    - 【法则3：稳接驳】使用 `types.MethodType` 绑定新的前向传播方法，防止 DDP 追踪计算图时 self 指针错乱。
    """

    def __init__(
            self,
            lora_r,
            lora_alpha,
            lora_dropout,
            lora_target_modules,
            gradient_checkpointing,
            visual_adapter_hidden_dim=4096,
            visual_adapter_r=16,
    ):
        # 语言端 LoRA 超参数
        self.r = lora_r
        self.alpha = lora_alpha
        self.dropout = lora_dropout
        self.target_modules = lora_target_modules
        self.gradient_checkpointing = gradient_checkpointing

        # 视觉端 Adapter 超参数
        self.visual_adapter_hidden_dim = visual_adapter_hidden_dim
        self.visual_adapter_r = visual_adapter_r

    def wrap(self, model: PreTrainedModel) -> PreTrainedModel:
        # =====================================================================
        # 1. 基础环境准备与梯度检查点 (Gradient Checkpointing) 配置
        # =====================================================================
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=self.gradient_checkpointing,
            # ⚠️ 极度关键：use_reentrant=False 能彻底解决 DDP 下梯度计算图断裂/重复标记的报错
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        # 开启梯度检查点时，必须关闭自回归缓存以节省显存并防止计算图冲突
        if hasattr(model, "config"):
            model.config.use_cache = False

        # =====================================================================
        # 2. 构造并注入 LLM 端的 LoRA
        # =====================================================================
        lora_config = LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            target_modules=self.target_modules,
            lora_dropout=self.dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(model, lora_config)

        # 强制要求输入层支持梯度反传，否则视觉特征传过来的梯度会在这里断掉
        if hasattr(peft_model, "enable_input_require_grads"):
            peft_model.enable_input_require_grads()

        # =====================================================================
        # 3. 👑 核心手术台：植入视觉残差适配器 (Visual Adapter)
        # =====================================================================
        print("\n✨ [Visual Adapter] 正在启动视觉塔脑机接驳手术...")

        # 3.1 极速定位视觉塔
        # 解析：原 model 被 get_peft_model 包装后，真身藏在 base_model.model 里。
        # Qwen3VL 的官方实现中，外层是 ForConditionalGeneration，内层才是含 visual 属性的 Model。
        vision_tower = peft_model.base_model.model.model.visual

        # 使用配置中硬编码的维度 (在 Qwen3-VL-8B 中测得为 4096)
        hidden_dim = self.visual_adapter_hidden_dim

        # 3.2 自动推断底座模型当前使用的 dtype 和 device
        ref_param = next(vision_tower.parameters())
        adapter_dtype = ref_param.dtype if ref_param.is_floating_point() else torch.bfloat16

        # 3.3 【法则 1：上户口】实例化并强行挂载为 vision_tower 的属性
        # 这样 HF Trainer 在扫描 requires_grad=True 的参数时，就能把它抓进 AdamW 优化器里
        vision_tower.res_adapter = Qwen3VLVisualAdapter(
            hidden_dim=hidden_dim,
            r=self.visual_adapter_r,
        ).to(device=ref_param.device, dtype=adapter_dtype)

        # 3.4 备份底座模型原生的前向传播逻辑
        vision_tower.original_forward = vision_tower.forward

        # 3.5 编写劫持函数：在原始视觉特征输出后，追加我们的 Adapter 逻辑
        def patched_forward(self, *args, **kwargs):
            # 先让 Qwen3-VL 提取原生的自然图像特征
            outputs = self.original_forward(*args, **kwargs)

            # ⚠️ Qwen3-VL 的视觉输出不是单一 Tensor，而是 BaseModelOutputWithDeepstackFeatures 结构化对象
            # 我们只提取已经对齐到 LLM 维度的核心特征进行医学适配，不破坏底层的多尺度计算

            # 替换主特征 (Pooler Output)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                outputs.pooler_output = self.res_adapter(outputs.pooler_output)

            # 替换深层特征堆栈 (Deepstack Features)
            if hasattr(outputs, "deepstack_features") and outputs.deepstack_features is not None:
                outputs.deepstack_features = [
                    self.res_adapter(x) for x in outputs.deepstack_features
                ]

            return outputs

        # 3.6 【法则 3：脑机接驳】绑定新方法到对象实例
        # 保证前向传播时，patched_forward 内部调用的 self 准确指向 vision_tower
        vision_tower.forward = types.MethodType(patched_forward, vision_tower)

        print(
            f"✅ [Visual Adapter] 成功挂载！"
            f" r={self.visual_adapter_r}, d={hidden_dim} "
        )

        # =====================================================================
        # 4. 打印最终的可训练参数信息
        # =====================================================================
        # 这里的统计会自动包含 LoRA 以及刚才“上户口”成功的 res_adapter
        peft_model.print_trainable_parameters()
        return peft_model