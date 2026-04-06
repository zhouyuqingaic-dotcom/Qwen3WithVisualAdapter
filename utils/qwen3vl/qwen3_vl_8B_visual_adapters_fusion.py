import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 🔧 独立工具函数：视觉 Token 物理隔离切分
# ==========================================
def visual_token_split(x: torch.Tensor, grid_thw: torch.Tensor) -> list[torch.Tensor]:
    """
    专门负责将 Qwen-VL 展平的长序列，根据 grid_thw 重新切分为独立的图片 Token 列表。
    恢复每张图片的伪 Batch 维度，彻底杜绝 1D 卷积时的跨图片边界污染。

    x: shape [1, N_total, D]
    grid_thw: shape [B, 3], 包含 [t, h, w]
    返回: List[torch.Tensor], 每个 Tensor shape 为 [1, N_i, D]
    """
    t = grid_thw[:, 0]
    h = grid_thw[:, 1]
    w = grid_thw[:, 2]

    # 精确计算深层特征中每张图片存活的 Token 数量 (Qwen-VL的2x2池化特性)
    tokens_per_image = [int(n) for n in (t * (h // 2) * (w // 2)).tolist()]

    # 严格的防呆校验
    if sum(tokens_per_image) != x.size(1):
        raise ValueError(
            f"❌ 视觉Token数量对齐失败！"
            f"grid_thw 算出的总数={sum(tokens_per_image)}, 但输入序列长度={x.size(1)}"
        )

    # 沿序列维度(dim=1)切断，返回独立张量列表
    return list(torch.split(x, tokens_per_image, dim=1))

# ==========================================
# 1. 动态路由融合适配器 (终极完全体)
# ==========================================
class Qwen3VLMoEVisualAdapterDynamicFusion(nn.Module):
    """
    动态多尺度混合专家适配器
    内部集成 Router MLP 与三大尺度专家。
    依赖外部传入的 BioMedCLIP 特征动态计算融合权重。
    """

    def __init__(self, hidden_dim: int,
                 adapter_global: nn.Module,
                 adapter_local: nn.Module,
                 adapter_region: nn.Module,
                 biomedclip_image_encoder_dim:int=512,
                 biomedclip_text_encoder_dim:int=512):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.norm = nn.LayerNorm(hidden_dim)

        # 实例化三大专家
        self.adapter_global = adapter_global
        self.adapter_local = adapter_local
        self.adapter_region = adapter_region

        # 实例化路由网络
        input_dim = biomedclip_image_encoder_dim + biomedclip_text_encoder_dim
        self.router_mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 3)
        )
        nn.init.zeros_(self.router_mlp[-1].weight)
        nn.init.zeros_(self.router_mlp[-1].bias)

    def forward(
            self,
            x: torch.Tensor,
            biomed_img_feat: torch.Tensor,
            biomed_txt_feat: torch.Tensor,
            grid_thw: torch.Tensor
    ) -> torch.Tensor:
        if biomed_img_feat is None or biomed_txt_feat is None:
            raise ValueError("⚠️ 动态融合模式下，必须传入 biomed_img_feat 和 biomed_txt_feat！")
        if grid_thw is None:
            raise ValueError("⚠️ 动态融合模式下，必须传入 grid_thw！")

        is_2d = (x.dim() == 2)
        if is_2d:
            x = x.unsqueeze(0)

        # 🚀 1. 算出每张图片的动态权重 [B, 3]
        combined_features = torch.cat([biomed_img_feat, biomed_txt_feat], dim=-1)
        router_logits = self.router_mlp(combined_features)
        router_weights = torch.softmax(router_logits, dim=-1)  # [B, 3]

        # 🚀 2. 进行物理隔离切分
        x_splits = visual_token_split(x, grid_thw)
        moe_residual_list = []

        # 🚀 3. 在循环内：归一化 -> 过网 -> 直接加权求和
        for i, sub_x in enumerate(x_splits):
            # 获取当前图片 i 的权重 (3 个标量)
            w_g, w_l, w_r = router_weights[i, 0], router_weights[i, 1], router_weights[i, 2]

            # 单张图独立 LayerNorm
            norm_sub_x = self.norm(sub_x)

            # 单张图独立过三大专家
            res_g = self.adapter_global(norm_sub_x)
            res_l = self.adapter_local(norm_sub_x)
            res_r = self.adapter_region(norm_sub_x)

            # 💡 极其优雅：直接在切分状态下，把权重乘进去，算好这张图的总残差
            sub_moe_residual = w_g * res_g + w_l * res_l + w_r * res_r
            moe_residual_list.append(sub_moe_residual)

        # 🚀 4. 只有算好的最终残差特征，才拼回成统一的长条
        moe_residual = torch.cat(moe_residual_list, dim=1)

        # 🚀 5. 最后经典的残差相加
        out = x + moe_residual

        if is_2d:
            out = out.squeeze(0)

        return out


# ==========================================
# 2. 静态硬融合适配器 (消融实验专用)
# ==========================================
class Qwen3VLMoEVisualAdapterFixedFusion(nn.Module):
    """
    静态多尺度混合专家适配器
    没有大脑，只根据初始化时传入的固定权重强行融合。
    """

    def __init__(self,
                 hidden_dim: int,
                 adapter_global: nn.Module,
                 adapter_local: nn.Module,
                 adapter_region: nn.Module,
                 fixed_weights: list[float] = [0.33, 0.33, 0.34]):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fixed_weights = fixed_weights

        self.norm = nn.LayerNorm(hidden_dim)

        self.adapter_global = adapter_global
        self.adapter_local = adapter_local
        self.adapter_region = adapter_region

    def forward(
            self,
            x: torch.Tensor,
            grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """
        接收 *args 和 **kwargs 是为了兼容 Wrapper 的统一调用。
        即使 Wrapper 传了 biomed_img_feat，这里也会安全地忽略它。
        """
        #  补上这句防呆校验，防止外层 Wrapper 漏传导致切分函数报错
        if grid_thw is None:
            raise ValueError("⚠️ Fixed融合模式下，必须传入 grid_thw！")

        is_2d = (x.dim() == 2)
        if is_2d:
            x = x.unsqueeze(0)

        # 🚀 1. 先进行物理隔离切分 (此时 x 还是原始特征)
        x_splits = visual_token_split(x, grid_thw)
        res_g_list, res_l_list, res_r_list = [], [], []

        # 🚀 2. 在每个子序列的循环内进行归一化和专家处理
        for sub_x in x_splits:
            # 针对单张图片的 Token 独立进行 LayerNorm
            norm_sub_x = self.norm(sub_x)

            # 喂给已经杜绝了边界污染的专家
            res_g_list.append(self.adapter_global(norm_sub_x))
            res_l_list.append(self.adapter_local(norm_sub_x))
            res_r_list.append(self.adapter_region(norm_sub_x))

        # 🚀 3. 重新拼接成长条
        res_g = torch.cat(res_g_list, dim=1)
        res_l = torch.cat(res_l_list, dim=1)
        res_r = torch.cat(res_r_list, dim=1)

        # 🚀 4. 融合输出
        w_g, w_l, w_r = self.fixed_weights
        fixed_alpha_residual = w_g * res_g + w_l * res_l + w_r * res_r
        out = x + fixed_alpha_residual  # 经典的 x + Adapter(Norm(x)) 结构

        if is_2d:
            out = out.squeeze(0)
        return out