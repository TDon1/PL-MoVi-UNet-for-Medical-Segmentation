import torch
import torch.nn as nn


class DCAF(nn.Module):
    """
    DCAF: Dual-stream Channel-spatial Adaptive Fusion

    双流通道-空间自适应融合模块，用于融合CNN和Transformer特征。

    Pipeline:
        1. Channel Attention - 双分支独立的通道注意力校准
        2. Spatial Attention - 联合空间关系建模
        3. Cross-Modal Interaction - 瓶颈式跨模态特征交互
        4. Dynamic Fusion - 内容自适应的动态权重融合

    Args:
        in_channels (int): 输入特征通道数
        reduction (int): 通道注意力降维比例，默认16

    Shape:
        - Input:
            - x_conv: CNN分支特征 [B, C, H, W]
            - x_transformer: Transformer分支特征 [B, C, H, W]
        - Output:
            - fused_feature: 融合特征 [B, C, H, W]
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super(DCAF, self).__init__()

        # ========== 通道注意力分支 ==========
        self.channel_attention = nn.ModuleDict({
            'avg_pool': nn.AdaptiveAvgPool2d(1),
            'max_pool': nn.AdaptiveMaxPool2d(1),
            'fc_conv': nn.Sequential(
                nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
            ),
            'fc_trans': nn.Sequential(
                nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
            )
        })

        # ========== 空间注意力分支 ==========
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(4, 2, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 2, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        # ========== 跨模态交互模块 ==========
        self.cross_modal = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1, bias=False)
        )

        # ========== 动态权重生成 ==========
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, 2, kernel_size=1, bias=False),
            nn.Softmax(dim=1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_conv: torch.Tensor, x_transformer: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x_conv: CNN分支特征 [B, C, H, W]
            x_transformer: Transformer分支特征 [B, C, H, W]

        Returns:
            fused_feature: 融合后的特征 [B, C, H, W]
        """
        B, C, H, W = x_conv.shape

        # ========== Step 1: 通道注意力 ==========
        # CNN 分支
        avg_conv = self.channel_attention['fc_conv'](
            self.channel_attention['avg_pool'](x_conv)
        )
        max_conv = self.channel_attention['fc_conv'](
            self.channel_attention['max_pool'](x_conv)
        )
        channel_attn_conv = self.sigmoid(avg_conv + max_conv)

        # Transformer 分支
        avg_trans = self.channel_attention['fc_trans'](
            self.channel_attention['avg_pool'](x_transformer)
        )
        max_trans = self.channel_attention['fc_trans'](
            self.channel_attention['max_pool'](x_transformer)
        )
        channel_attn_trans = self.sigmoid(avg_trans + max_trans)

        # 通道注意力加权
        x_conv_ca = x_conv * channel_attn_conv
        x_trans_ca = x_transformer * channel_attn_trans

        # ========== Step 2: 空间注意力 ==========
        # 生成空间注意力的输入（通道维度的统计信息）
        spatial_input = torch.cat([
            torch.mean(x_conv, dim=1, keepdim=True),
            torch.max(x_conv, dim=1, keepdim=True)[0],
            torch.mean(x_transformer, dim=1, keepdim=True),
            torch.max(x_transformer, dim=1, keepdim=True)[0]
        ], dim=1)  # [B, 4, H, W]

        spatial_attn = self.spatial_attention(spatial_input)  # [B, 2, H, W]
        spatial_attn_conv, spatial_attn_trans = torch.chunk(spatial_attn, 2, dim=1)

        # 空间注意力加权
        x_conv_sa = x_conv_ca * spatial_attn_conv
        x_trans_sa = x_trans_ca * spatial_attn_trans

        # ========== Step 3: 跨模态交互 ==========
        concat_features = torch.cat([x_conv_sa, x_trans_sa], dim=1)
        cross_modal_features = self.cross_modal(concat_features)
        x_conv_cross, x_trans_cross = torch.chunk(cross_modal_features, 2, dim=1)

        # ========== Step 4: 动态权重融合 ==========
        # 计算动态融合权重
        combined = torch.cat([x_conv_cross, x_trans_cross], dim=1)
        dynamic_weights = self.weight_generator(combined)  # [B, 2, 1, 1]
        weight_conv = dynamic_weights[:, 0:1, :, :]
        weight_trans = dynamic_weights[:, 1:2, :, :]

        # 最终融合
        fused_feature = x_conv_cross * weight_conv + x_trans_cross * weight_trans

        return fused_feature


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 创建模块实例
    dcaf = DCAF(in_channels=256, reduction=16)

    # 生成测试输入
    x_cnn = torch.randn(2, 256, 32, 32)
    x_trans = torch.randn(2, 256, 32, 32)

    # 前向传播
    output = dcaf(x_cnn, x_trans)

    print(f"Input CNN shape: {x_cnn.shape}")
    print(f"Input Transformer shape: {x_trans.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in dcaf.parameters()):,}")
