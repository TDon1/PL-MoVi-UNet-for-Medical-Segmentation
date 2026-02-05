import torch
from torch import nn
from torch.nn import functional as F


class MultiLevelSpatialPyramid(nn.Module):
    """
    MLSP: Multi-Level Spatial Pyramid

    多级空间金字塔模块，通过分层密集采样实现多尺度特征提取，专为小目标分割优化。

    Architecture:
        - Level 1 (Fine-grained): 1×1, 3×3, dilation=[2,3] → 捕获细粒度局部特征
        - Level 2 (Medium-range): dilation=[6,12] → 编码中等感受野上下文
        - Level 3 (Global): AdaptiveAvgPool(1×1) → 聚合全局语义信息
        - Level 4 (Local Pyramid): Pool[8×8, 4×4, 2×2] → 保留多分辨率空间结构

    总分支数: 10 (4 fine + 2 medium + 1 global + 3 local pyramid)

    Args:
        in_channels (int): 输入特征通道数
        out_channels (int): 输出特征通道数

    Shape:
        - Input: (B, in_channels, H, W)
        - Output: (B, out_channels, H, W)

    Example:
        >>> mlsp = MultiLevelSpatialPyramid(256, 256)
        >>> x = torch.randn(2, 256, 64, 64)
        >>> out = mlsp(x)  # (2, 256, 64, 64)
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(MultiLevelSpatialPyramid, self).__init__()

        # ============ Level 1: 细粒度小尺度分支 (专门针对小目标) ============
        # 使用密集的小感受野捕获精细局部特征
        self.small_scale = nn.ModuleList([
            # Branch 1: 1×1 卷积 - 点级特征，感受野 1×1
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(8, out_channels),
                nn.ReLU(inplace=True)
            ),
            # Branch 2: 3×3 标准卷积 - 最小感受野 3×3
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.GroupNorm(8, out_channels),
                nn.ReLU(inplace=True)
            ),
            # Branch 3: 空洞率=2 - 感受野 5×5，捕获小目标周边上下文
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2, bias=False),
                nn.GroupNorm(8, out_channels),
                nn.ReLU(inplace=True)
            ),
            # Branch 4: 空洞率=3 - 感受野 7×7，进一步扩展局部感受野
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=3, dilation=3, bias=False),
                nn.GroupNorm(8, out_channels),
                nn.ReLU(inplace=True)
            ),
        ])

        # ============ Level 2: 中等尺度上下文分支 ============
        # 捕获中等范围的空间关系和语义信息
        self.medium_scale = nn.ModuleList([
            # Branch 5: 空洞率=6 - 感受野 13×13
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
                nn.GroupNorm(8, out_channels),
                nn.ReLU(inplace=True)
            ),
            # Branch 6: 空洞率=12 - 感受野 25×25
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
                nn.GroupNorm(8, out_channels),
                nn.ReLU(inplace=True)
            ),
        ])

        # ============ Level 3: 全局语义分支 ============
        # Branch 7: 全局平均池化 + 1×1卷积，编码图像级全局上下文
        self.large_scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 压缩为 1×1
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )

        # ============ Level 4: 局部金字塔池化分支 ============
        # 多分辨率池化保留更多空间信息，避免小目标被完全池化丢失
        self.local_pools = nn.ModuleList([
            # Branch 8: 8×8 池化 - 保留较多空间细节
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(8, 8)),
                nn.Conv2d(in_channels, out_channels // 4, 1, bias=False),
                nn.ReLU(inplace=True)
            ),
            # Branch 9: 4×4 池化 - 中等空间抽象
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(4, 4)),
                nn.Conv2d(in_channels, out_channels // 4, 1, bias=False),
                nn.ReLU(inplace=True)
            ),
            # Branch 10: 2×2 池化 - 高度抽象的空间信息
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(2, 2)),
                nn.Conv2d(in_channels, out_channels // 4, 1, bias=False),
                nn.ReLU(inplace=True)
            ),
        ])

        # ============ 计算融合层输入通道数 ============
        # 总通道数计算:
        #   Level 1 (small_scale): 4 × out_channels
        #   Level 2 (medium_scale): 2 × out_channels
        #   Level 3 (large_scale): 1 × out_channels
        #   Level 4 (local_pools): 3 × (out_channels // 4)
        # 总计 = 4C + 2C + C + 3×(C/4) = 7C + 0.75C = 7.75C
        total_channels = (
                out_channels * 4 +  # Level 1
                out_channels * 2 +  # Level 2
                out_channels +  # Level 3
                (out_channels // 4) * 3  # Level 4
        )

        # ============ 自适应特征融合层 ============
        # 将所有分支的特征融合为统一的输出特征
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, 1, bias=False),  # 1×1卷积降维
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)  # 防止过拟合
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x (torch.Tensor): 输入特征 [B, in_channels, H, W]

        Returns:
            torch.Tensor: 融合后的多尺度特征 [B, out_channels, H, W]
        """
        # 记录原始空间尺寸，用于后续上采样对齐
        size = x.shape[-2:]
        features = []

        # ============ Level 1: 提取细粒度小尺度特征 ============
        for conv in self.small_scale:
            features.append(conv(x))

        # ============ Level 2: 提取中等尺度上下文特征 ============
        for conv in self.medium_scale:
            features.append(conv(x))

        # ============ Level 3: 提取全局语义特征 ============
        global_feat = self.large_scale(x)
        # 上采样至原始尺寸 [B, C, 1, 1] → [B, C, H, W]
        global_feat = F.interpolate(
            global_feat,
            size=size,
            mode='bilinear',
            align_corners=False
        )
        features.append(global_feat)

        # ============ Level 4: 提取局部金字塔特征 ============
        for pool in self.local_pools:
            local_feat = pool(x)
            # 上采样至原始尺寸 [B, C/4, pool_size, pool_size] → [B, C/4, H, W]
            local_feat = F.interpolate(
                local_feat,
                size=size,
                mode='bilinear',
                align_corners=False
            )
            features.append(local_feat)

        # ============ 多级特征融合 ============
        # 在通道维度拼接所有特征 [B, total_channels, H, W]
        out = torch.cat(features, dim=1)

        # 通过融合层生成最终输出 [B, out_channels, H, W]
        out = self.fusion(out)

        return out


# ============ 使用示例 ============
if __name__ == "__main__":
    # 创建模块实例
    mlsp = MultiLevelSpatialPyramid(in_channels=256, out_channels=256)

    # 生成测试输入
    x = torch.randn(2, 256, 64, 64)

    # 前向传播
    output = mlsp(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in mlsp.parameters()):,}")
