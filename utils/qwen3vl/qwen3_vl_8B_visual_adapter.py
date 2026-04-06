import torch
import torch.nn as nn


class BasicVisualAdapter(nn.Module):
    """
    基本视觉适配器 (Single-Scale Expert Adapter)
    通过改变 kernel_size 和 dilation 来控制该Adapter的“医学视野(感受野)”。
    """

    def __init__(self, hidden_dim: int, r: int = 16, kernel_size: int = 1, dilation: int = 1):
        super().__init__()
        self.down = nn.Linear(hidden_dim, hidden_dim // r, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden_dim // r, hidden_dim, bias=False)

        self.use_conv = (kernel_size > 1)
        if self.use_conv:
            # 动态计算 padding，保证 1D 卷积后序列长度严格不变
            padding = (kernel_size - 1) * dilation // 2
            self.conv = nn.Conv1d(
                in_channels=hidden_dim // r,
                out_channels=hidden_dim // r,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=hidden_dim // r,  # 深度可分离，极致轻量
                bias=False
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.use_conv:
            nn.init.kaiming_uniform_(self.conv.weight, a=torch.math.sqrt(5))
        nn.init.zeros_(self.up.weight)  # 严格零初始化，防止初期 Loss 爆炸

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Seq_len, Hidden_dim]
        h = self.down(x)
        if self.use_conv:
            h = h.transpose(-1, -2)  # [Batch, Channels, Seq]
            h = self.conv(h)
            h = h.transpose(-1, -2)
        return self.up(self.act(h))


class VisualAdapter_Global(BasicVisualAdapter):
    def __init__(self, hidden_dim: int, r: int = 16, kernel_size: int = 1, dilation: int = 1):
        super().__init__(hidden_dim=hidden_dim,r=r,kernel_size=kernel_size,dilation=dilation)


class VisualAdapter_Local(BasicVisualAdapter):
    def __init__(self, hidden_dim: int, r: int = 16, kernel_size: int = 3, dilation: int = 1):
        super().__init__(hidden_dim=hidden_dim,r=r,kernel_size=kernel_size,dilation=dilation)


class VisualAdapter_Region(BasicVisualAdapter):
    def __init__(self,hidden_dim: int, r: int = 16, kernel_size: int = 5, dilation: int = 1):
        super().__init__(hidden_dim=hidden_dim,r=r,kernel_size=kernel_size,dilation=dilation)


