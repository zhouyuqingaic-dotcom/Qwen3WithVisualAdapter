import torch
import torch.nn as nn


class Qwen3VLVisualAdapter(nn.Module):
    """
    Qwen3-VL 专属的视觉残差适配器 (Visual Residual Adapter)。

    设计极度轻量：LayerNorm -> Linear(down) -> GELU -> Linear(up)
    核心特性：最后一层 Linear(up) 零初始化，确保初始时刻网络处于“恒等映射” (Identity Mapping) 状态，
    避免破坏预训练视觉特征导致初始 Loss 爆炸。
    """

    def __init__(self, hidden_dim: int, r: int = 16, init_alpha: float = 0.1):
        """
        参数说明:
        - hidden_dim: 视觉塔输出特征的维度 (例如 Qwen-VL 中通常是 3584)
        - r: 瓶颈层的压缩倍率 (Reduction factor)，控制 Adapter 的参数量
        - init_alpha: 残差连接的可学习缩放因子的初始值
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.r = r

        # 1. 归一化层，保证输入瓶颈层的特征分布稳定
        self.norm = nn.LayerNorm(hidden_dim)

        # 2. 降维投影层 (Bottleneck Down)
        # 不使用 bias 可以进一步减少参数量，符合轻量化微调的原则
        self.down = nn.Linear(hidden_dim, hidden_dim // r, bias=False)

        # 3. 激活函数
        self.act = nn.GELU()

        # 4. 升维投影层 (Bottleneck Up)
        self.up = nn.Linear(hidden_dim // r, hidden_dim, bias=False)

        # 5. 可学习的残差缩放因子
        # 初始给一个较小的值 (如 0.1)，让模型在前期主要依赖原始特征，慢慢学会利用新特征
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

        # ✨ 执行极其关键的零初始化
        self._reset_parameters()

    def _reset_parameters(self):
        """
        安全初始化机制。
        下采样层 (down) 保持 PyTorch 默认的 Kaiming 均匀分布初始化即可，
        但上采样层 (up) 的权重必须严格清零！
        """
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        x 的 shape 通常为 [batch_size, sequence_length, hidden_dim]
        """
        # 1. 提取医学视觉残差
        residual = self.up(self.act(self.down(self.norm(x))))

        # 2. 与原始自然视觉特征相加
        # 训练第 0 步时，由于 self.up 权重全为 0，residual 必然全为 0，
        # 所以 return 的结果 100% 等价于原始输入 x。
        return x + self.alpha * residual