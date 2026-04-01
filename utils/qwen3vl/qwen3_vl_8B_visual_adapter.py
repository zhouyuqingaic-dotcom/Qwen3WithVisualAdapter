import torch
import torch.nn as nn


class Qwen3VLVisualAdapter(nn.Module):
    """
    Qwen3-VL 进阶版视觉适配器：Spatial-Aware Routing Adapter (SARA)

    【核心创新与物理意义】
    1. Depthwise Conv1D (空间感知):
       传统的 Linear 瓶颈层缺乏空间归纳偏置。这里引入了 1D 深度可分离卷积，
       让相邻的图像 Token (Patch) 能够进行局部特征交互，极大增强了模型对微小医学病灶的定位能力。

    2. Dynamic Gating (内容感知解耦与路由):
       摒弃了传统 Vanilla Adapter 全局污染式的静态 alpha。
       引入基于 Token 的动态 Sigmoid 门控。对于正常的背景组织，门控趋于 0，完美保护底座 VLM 的宏观常识；
       对于复杂的病理组织，门控趋于 1，精准注入深度的医学残差特征。
       从而完美解决 Open-ended(复杂推理) 和 Closed-ended(简单判别) 任务的性能跷跷板效应。
    """

    def __init__(self, hidden_dim: int, r: int = 16):
        """
        参数说明:
        - hidden_dim: 视觉塔输出特征的维度 (Qwen3-VL 中测得为 4096)
        - r: 瓶颈层的压缩倍率 (Reduction factor)，控制 Adapter 的参数量
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.r = r

        # 1. 归一化层，稳定输入分布
        self.norm = nn.LayerNorm(hidden_dim)

        # 2. 降维投影 (Bottleneck Down)
        self.down = nn.Linear(hidden_dim, hidden_dim // r, bias=False)

        # 3. 空间感知层：序列级的 1D 深度可分离卷积 (Depthwise Conv1D)
        # padding=1 保证输入输出 sequence length 严格不变，完美适配 Qwen 的动态分辨率
        self.conv = nn.Conv1d(
            in_channels=hidden_dim // r,
            out_channels=hidden_dim // r,
            kernel_size=3,
            padding=1,
            groups=hidden_dim // r,  # 分组卷积，参数量极小，只做空间混合不做通道混合
            bias=False
        )

        # 4. 激活函数
        self.act = nn.GELU()

        # 5. 升维投影 (Bottleneck Up)
        self.up = nn.Linear(hidden_dim // r, hidden_dim, bias=False)

        # ✨ 6. 核心创新：动态门控层 (Dynamic Gate)
        # 针对每个 Token 输出一个 0~1 的路由权重
        self.gate = nn.Linear(hidden_dim, 1)

        # 执行关键的防崩塌初始化
        self._reset_parameters()

    def _reset_parameters(self):
        """极其关键的安全初始化"""
        # 卷积层使用 Kaiming 初始化
        nn.init.kaiming_uniform_(self.conv.weight, a=torch.math.sqrt(5))

        # 升维层严格清零，保证初始训练阶段残差输出为 0 (Identity Mapping)
        nn.init.zeros_(self.up.weight)

        # 门控层初始化：设置 Bias = -2.197，使得初始的 Sigmoid(Gate) 约等于 0.1
        # 这让训练初期的行为与 Vanilla Adapter 类似，防止初期梯度剧烈震荡
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.197)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        x shape: [batch_size, seq_len, hidden_dim]
        """
        # 🛡️ 核心修复：自动探测并记录维度
        is_2d = (x.dim() == 2)

        # 如果是 2D 张量 [Total_Seq_Len, Hidden_dim]，临时升维成 [1, Total_Seq_Len, Hidden_dim]
        if is_2d:
            x = x.unsqueeze(0)

        # 1. 降维
        hidden = self.down(self.norm(x))

        # 2. 局部空间交互 (Conv1d 要求通道维在中间: [Batch, Channels, Seq_len])
        hidden = hidden.transpose(1, 2)
        hidden = self.conv(hidden)
        hidden = hidden.transpose(1, 2)

        # 3. 激活与升维
        residual = self.up(self.act(hidden))

        # 4. 提取基于视觉内容的动态门控权重
        # gate_score shape: [batch_size, seq_len, 1]
        gate_score = torch.sigmoid(self.gate(x))

        # 5. 动态融合：底座原生特征 + (门控热力图 * 医学残差特征)
        out = x + gate_score * residual

        # 🛡️ 核心修复：如果进来时是 2D，出去时也必须脱掉 Batch 维还给底座
        if is_2d:
            out = out.squeeze(0)

        return out