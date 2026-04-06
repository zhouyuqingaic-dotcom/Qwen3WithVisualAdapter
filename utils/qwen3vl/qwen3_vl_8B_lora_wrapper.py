import types
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import PreTrainedModel

# 导入我们刚刚写好的底层兵器库与指挥部
from utils.qwen3vl.qwen3_vl_8B_visual_adapter import VisualAdapter_Global, VisualAdapter_Local, VisualAdapter_Region
from utils.qwen3vl.qwen3_vl_8B_visual_adapters_fusion import Qwen3VLMoEVisualAdapterDynamicFusion, \
    Qwen3VLMoEVisualAdapterFixedFusion


# =====================================================================
# 🧠 新增：冻结的 BioMedCLIP 跨模态特征提取器
# =====================================================================
class FrozenBioMedCLIPFeatureExtractor(nn.Module):
    def __init__(self, raw_model):
        super().__init__()
        self.model = raw_model
        self.model.eval()
        # 绝对冻结，不参与任何梯度更新
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, biomed_img_tensors, biomed_txt_tokens):
        with torch.no_grad():
            img_feat, txt_feat, _ = self.model(biomed_img_tensors, biomed_txt_tokens)
            img_feat = torch.nn.functional.normalize(img_feat, dim=-1)
            txt_feat = torch.nn.functional.normalize(txt_feat, dim=-1)
        return img_feat, txt_feat


# =====================================================================
# 👑 核心重构：多尺度混合专家 Wrapper (Dual-Tuning MoE)
# =====================================================================
class Qwen3VLLoraAndVisualAdapterWrapper:
    def __init__(
            self,
            lora_r, lora_alpha, lora_dropout, lora_target_modules, gradient_checkpointing,
            visual_adapter_hidden_dim=4096,
            visual_adapter_r=16,
            router_mode="dynamic",
            biomed_extractor=None,
            global_adapter_kernel_size=1,
            local_adapter_kernel_size=3,
            region_adapter_kernel_size=5,
            fixed_weights=[0.33, 0.33, 0.34]  # 👈 【修复 1】接住 Trainer 传来的 fixed_weights
    ):
        self.r = lora_r
        self.alpha = lora_alpha
        self.dropout = lora_dropout
        self.target_modules = lora_target_modules
        self.gradient_checkpointing = gradient_checkpointing
        self.visual_adapter_hidden_dim = visual_adapter_hidden_dim
        self.visual_adapter_r = visual_adapter_r
        self.router_mode = router_mode
        self.biomed_extractor = biomed_extractor

        self.global_adapter_kernel_size = global_adapter_kernel_size
        self.local_adapter_kernel_size = local_adapter_kernel_size
        self.region_adapter_kernel_size = region_adapter_kernel_size
        self.fixed_weights = fixed_weights  # 👈 【修复 1】保存为类属性

    def wrap(self, model: PreTrainedModel) -> PreTrainedModel:
        # 1. 基础 LoRA 环境准备
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=self.gradient_checkpointing,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "config"): model.config.use_cache = False
        peft_model = get_peft_model(model,
                                    LoraConfig(r=self.r, lora_alpha=self.alpha, target_modules=self.target_modules,
                                               lora_dropout=self.dropout, bias="none", task_type="CAUSAL_LM"))
        if hasattr(peft_model, "enable_input_require_grads"): peft_model.enable_input_require_grads()

        # 2. 定位视觉塔并获取设备与精度
        print(f"\n✨ [MoE Adapter] 正在启动多尺度视觉塔脑机接驳手术 (模式: {self.router_mode})...")
        vision_tower = peft_model.base_model.model.model.visual
        hidden_dim = self.visual_adapter_hidden_dim
        r = self.visual_adapter_r
        ref_param = next(vision_tower.parameters())
        adapter_dtype = ref_param.dtype if ref_param.is_floating_point() else torch.bfloat16

        # ===============================================================
        # 3. 【依赖注入】实例化三大视觉Adapter与融合枢纽
        # ===============================================================
        adapter_global = VisualAdapter_Global(hidden_dim=hidden_dim,
                                              r=r,
                                              kernel_size=self.global_adapter_kernel_size)
        adapter_local = VisualAdapter_Local(hidden_dim=hidden_dim,
                                            r=r,
                                            kernel_size=self.local_adapter_kernel_size)
        adapter_region = VisualAdapter_Region(hidden_dim=hidden_dim,
                                              r=r,
                                              kernel_size=self.region_adapter_kernel_size)

        if self.router_mode == "dynamic":
            #  【修复 2】检查 biomed_extractor 实例是否存在，而不是检查被淘汰的路径
            if self.biomed_extractor is None:
                raise ValueError("⚠️ 动态路由模式下，Trainer 必须传入实例化的 biomed_extractor!")

            fusion_layer = Qwen3VLMoEVisualAdapterDynamicFusion(
                hidden_dim=hidden_dim,
                adapter_global=adapter_global,
                adapter_local=adapter_local,
                adapter_region=adapter_region
            )
            # 挂载冻结的 BioMedCLIP 大脑到 peft_model 上
            peft_model.biomed_extractor = self.biomed_extractor.to(device=ref_param.device, dtype=adapter_dtype)
        else:
            fusion_layer = Qwen3VLMoEVisualAdapterFixedFusion(
                hidden_dim=hidden_dim,
                adapter_global=adapter_global,
                adapter_local=adapter_local,
                adapter_region=adapter_region,
                fixed_weights=self.fixed_weights  # 👈 【修复 3】把外面的硬融合比例准确无误地传给底座！
            )

        # 上户口：将融合枢纽挂载到视觉塔，使其被 PyTorch 追踪并更新梯度
        vision_tower.res_adapter = fusion_layer.to(device=ref_param.device, dtype=adapter_dtype)

        # ===============================================================
        # 4. 【第一重劫持】外层大脑截获与传送 (Outermost Forward Patch)
        # ===============================================================
        peft_model.original_forward = peft_model.forward

        def patched_model_forward(self, *args, **kwargs):
            # 💡 极度关键的 kwargs.pop()：
            biomed_img = kwargs.pop("biomed_image_tensors", None)
            biomed_txt = kwargs.pop("biomed_text_tokens", None)

            # 只有在动态模式且传了参数的情况下，才激活大脑
            if biomed_img is not None and biomed_txt is not None and hasattr(self, "biomed_extractor"):
                img_f, txt_f = self.biomed_extractor(biomed_img, biomed_txt)
                # 🛸 时空传送：把算出的跨模态特征强行塞进视觉塔的隐式变量里
                self.base_model.model.model.visual.current_biomed_img_feat = img_f
                self.base_model.model.model.visual.current_biomed_txt_feat = txt_f

            # 干净利落地调用原始大模型前向传播
            return self.original_forward(*args, **kwargs)

        peft_model.forward = types.MethodType(patched_model_forward, peft_model)

        # ===============================================================
        # 5. 【第二重劫持】内层肌肉接收与融合 (Inner Vision Tower Patch)
        # ===============================================================
        vision_tower.original_forward = vision_tower.forward

        def patched_vision_forward(self, *args, **kwargs):
            outputs = self.original_forward(*args, **kwargs)

            # 接收外层传送过来的指令
            img_f = getattr(self, "current_biomed_img_feat", None)
            txt_f = getattr(self, "current_biomed_txt_feat", None)

            # 📦 【绝杀提取】：从 Qwen 底层参数中获取图像的 3D 网格尺寸 (grid_thw)
            grid_thw = kwargs.get("grid_thw", None)
            if grid_thw is None and len(args) > 1:
                grid_thw = args[1]

            # 替换特征，并将特征传给 Fusion 层 (⚠️ 注意这里补上了 grid_thw)
            # ==========================================================
            # 🚀 替换特征，并将特征传给 Fusion 层
            # ==========================================================
            #动态模式传了 4 个参数，静态模式只传了 2 个参数。
            is_dynamic = (img_f is not None and txt_f is not None)

            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                if is_dynamic:
                    outputs.pooler_output = self.res_adapter(
                        outputs.pooler_output,
                        biomed_img_feat=img_f,
                        biomed_txt_feat=txt_f,
                        grid_thw=grid_thw
                    )
                else:
                    outputs.pooler_output = self.res_adapter(
                        outputs.pooler_output,
                        grid_thw=grid_thw
                    )

            if hasattr(outputs, "deepstack_features") and outputs.deepstack_features is not None:
                if is_dynamic:
                    outputs.deepstack_features = [
                        self.res_adapter(
                            x,
                            biomed_img_feat=img_f,
                            biomed_txt_feat=txt_f,
                            grid_thw=grid_thw
                        ) for x in outputs.deepstack_features
                    ]
                else:
                    outputs.deepstack_features = [
                        self.res_adapter(
                            x,
                            grid_thw=grid_thw
                        ) for x in outputs.deepstack_features
                    ]
            return outputs

        vision_tower.forward = types.MethodType(patched_vision_forward, vision_tower)

        print(f"✅ [MoE Adapter] 成功挂载！融合模式: {self.router_mode.upper()}")
        peft_model.print_trainable_parameters()
        return peft_model

