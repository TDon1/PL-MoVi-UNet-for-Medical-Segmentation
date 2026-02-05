from torchinfo import summary
from collections import OrderedDict
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# 使用你上传的文件中的 mobilenet_v3_large
from src.plmovitunet.mobilev3 import mobilenet_v3_small
from src.plmovitunet.mobilev3 import mobilenet_v3_large
from src.plmovitunet.mobilevit import MobileViTBlock
# 使用你上传的文件中的 mobile_vit_xx_small
from src.plmovitunet.mobilevit import mobile_vit_small as mobilevit_large
from src.plmovitunet.mobilevit import mobile_vit_xx_small as mobilevit_small
from src.plmovitunet.MLSP import MultiLevelSpatialPyramid as MLSP
from src.plmovitunet.DCAF import DCAF


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out



class PLMoViTUnet_small(nn.Module):
    def __init__(self, num_classes,
                 pretrain_backbone: bool = False,
                 use_vit: bool = True,      # 控制是否使用 ViT 分支
                 use_dcaf: bool = True,     # 控制是否使用 DCAF 注意力融合
                 use_mlsp: bool = True):   # 控制是否使用 MLSP
        super(PLMoViTUnet_small, self).__init__()

        self.use_vit = use_vit
        self.use_dcaf = use_dcaf
        self.use_mlsp = use_mlsp

        # ---------------------------
        # 1. 构建 CNN Backbone
        # ---------------------------
        backbone_cnn = mobilenet_v3_small()
        backbone_cnn = backbone_cnn.features

        stage_cnn_indices = [0, 1, 3, 8, 11]
        self.stage_cnn_out_channels = [backbone_cnn[i].out_channels for i in stage_cnn_indices]
        return_cnnlayers = dict([(str(j), f"stage{i}") for i, j in enumerate(stage_cnn_indices)])
        self.backbone_cnn = IntermediateLayerGetter(backbone_cnn, return_layers=return_cnnlayers)

        # ---------------------------
        # 2. 构建 ViT Backbone (仅当 use_vit=True)
        # ---------------------------
        self.stage_vit_out_channels = []
        if self.use_vit:
            backbone_vit = mobilevit_small()
            stage_vit_indices = ["layer_1", "layer_2", "layer_3", "layer_4", "layer_5"]

            for layer in stage_vit_indices:
                layer_module = getattr(backbone_vit, layer)
                if isinstance(layer_module[-1], MobileViTBlock):
                    out_channels = layer_module[-1].cnn_in_dim
                elif hasattr(layer_module[-1], "out_channels"):
                    out_channels = layer_module[-1].out_channels
                else:
                    raise AttributeError(f"Cannot determine output channels for layer: {layer}")
                self.stage_vit_out_channels.append(out_channels)

            return_vitlayers = dict([(layer, f"stage{i}") for i, layer in enumerate(stage_vit_indices)])
            self.backbone_vit = IntermediateLayerGetter(backbone_vit, return_layers=return_vitlayers)

        # ---------------------------
        # 3. 构建融合模块 (Fusion)
        # ---------------------------
        self.fusion_layers = nn.ModuleList()

        for i in range(5):
            if not self.use_vit:
                # Row 1: 仅 CNN，无融合操作，占位符
                self.fusion_layers.append(nn.Identity())
            else:
                # 存在 ViT 分支
                if self.use_dcaf:
                    # Row 3 & 4: 使用 dcaf 注意力融合
                    self.fusion_layers.append(
                        DCAF(in_channels=self.stage_vit_out_channels[i])
                    )
                else:
                    # Row 2: 简单相加 (Simple Add)
                    # 如果 ViT 和 CNN 通道数不一致，需要 1x1 卷积对齐
                    if self.stage_vit_out_channels[i] != self.stage_cnn_out_channels[i]:
                        self.fusion_layers.append(
                            ConvAlign(self.stage_vit_out_channels[i], self.stage_cnn_out_channels[i])
                        )
                    else:
                        self.fusion_layers.append(nn.Identity())

        # ---------------------------
        # 4. 构建解码器 (Decoder / UpSampling)
        # ---------------------------
        c = self.stage_cnn_out_channels[4] + self.stage_cnn_out_channels[3]
        self.up1 = Up(c, self.stage_cnn_out_channels[3])

        c = self.stage_cnn_out_channels[3] + self.stage_cnn_out_channels[2]
        self.up2 = Up(c, self.stage_cnn_out_channels[2])

        c = self.stage_cnn_out_channels[2] + self.stage_cnn_out_channels[1]
        self.up3 = Up(c, self.stage_cnn_out_channels[1])

        c = self.stage_cnn_out_channels[1] + self.stage_cnn_out_channels[0]
        self.up4 = Up(c, self.stage_cnn_out_channels[0])

        # ---------------------------
        # 5. 可选组件 (MLSP)
        # ---------------------------
        if self.use_mlsp:
            # MLSP 应用在编码器末端 x4
            self.use_mlsp = MLSP(self.stage_cnn_out_channels[4], self.stage_cnn_out_channels[4])

        # ---------------------------
        # 6. 输出卷积
        # ---------------------------
        self.conv = OutConv(self.stage_cnn_out_channels[0], num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_shape = x.shape[-2:]

        # 1. 提取 CNN 特征
        backbone_cnnout = self.backbone_cnn(x)

        # 2. 提取 ViT 特征 (如果有)
        backbone_vitout = None
        if self.use_vit:
            backbone_vitout = self.backbone_vit(x)

        # 3. 特征融合循环
        features = []
        for i in range(5):
            stage_name = f'stage{i}'
            f_cnn = backbone_cnnout[stage_name]

            if not self.use_vit:
                # Row 1: 仅使用 CNN 特征
                x_fused = f_cnn
            else:
                f_vit = backbone_vitout[stage_name]

                if self.use_dcaf:
                    # Row 3 & 4: dcaf 融合
                    x_fused = self.fusion_layers[i](f_cnn, f_vit)
                else:
                    # Row 2: 简单相加
                    f_vit_aligned = self.fusion_layers[i](f_vit)
                    x_fused = f_cnn + f_vit_aligned

            features.append(x_fused)

        # 解包融合后的特征
        x0, x1, x2, x3, x4 = features[0], features[1], features[2], features[3], features[4]

        # 4. 在编码器末端应用 MLSP (如果有)
        if self.use_mlsp:
            x4 = self.use_mlsp(x4)

        # 5. 解码上采样
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

        # 6. 输出头
        x = self.conv(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        return x



# 如果需要简单的 1x1 卷积做维度对齐（用于实验2：无dcaf的简单融合）
class ConvAlign(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

#
# class IntermediateLayerGetter(nn.ModuleDict):
#     """
#     Module wrapper that returns intermediate layers from a model
#     """
#     _version = 2
#     __annotations__ = {
#         "return_layers": Dict[str, str],
#     }
#
#     def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
#         if not set(return_layers).issubset([name for name, _ in model.named_children()]):
#             raise ValueError("return_layers are not present in model")
#         orig_return_layers = return_layers
#         return_layers = {str(k): str(v) for k, v in return_layers.items()}
#
#         layers = OrderedDict()
#         for name, module in model.named_children():
#             layers[name] = module
#             if name in return_layers:
#                 del return_layers[name]
#             if not return_layers:
#                 break
#
#         super(IntermediateLayerGetter, self).__init__(layers)
#         self.return_layers = orig_return_layers
#
#     def forward(self, x: Tensor) -> Dict[str, Tensor]:
#         out = OrderedDict()
#         for name, module in self.items():
#             x = module(x)
#             if name in self.return_layers:
#                 out_name = self.return_layers[name]
#                 out[out_name] = x
#         return out




class PLMoViTUnet_large(nn.Module):
    def __init__(self, num_classes,
                 pretrain_backbone: bool = False,
                 use_vit: bool = True,  # 新增：控制是否使用 ViT 分支
                 use_dcaf: bool = True,  # 新增：控制是否使用  注意力融合
                 use_mlsp: bool = True):  # 控制是否使用 MLSP
        super(PLMoViTUnet_large, self).__init__()

        self.use_vit = use_vit
        self.use_dcaf = use_dcaf
        self.use_mlsp = use_mlsp

        # ---------------------------
        # 1. 构建 CNN Backbone
        # ---------------------------
        backbone_cnn = mobilenet_v3_large()
        backbone_cnn = backbone_cnn.features

        stage_cnn_indices = [1, 3, 6, 12, 15]
        self.stage_cnn_out_channels = [backbone_cnn[i].out_channels for i in stage_cnn_indices]
        return_cnnlayers = dict([(str(j), f"stage{i}") for i, j in enumerate(stage_cnn_indices)])
        self.backbone_cnn = IntermediateLayerGetter(backbone_cnn, return_layers=return_cnnlayers)

        # ---------------------------
        # 2. 构建 ViT Backbone (仅当 use_vit=True)
        # ---------------------------
        self.stage_vit_out_channels = []
        if self.use_vit:
            backbone_vit = mobilevit_large()
            stage_vit_indices = ["layer_1", "layer_2", "layer_3", "layer_4", "layer_5"]

            for layer in stage_vit_indices:
                layer_module = getattr(backbone_vit, layer)
                if isinstance(layer_module[-1], MobileViTBlock):
                    out_channels = layer_module[-1].cnn_in_dim
                elif hasattr(layer_module[-1], "out_channels"):
                    out_channels = layer_module[-1].out_channels
                else:
                    raise AttributeError(f"Cannot determine output channels for layer: {layer}")
                self.stage_vit_out_channels.append(out_channels)

            return_vitlayers = dict([(layer, f"stage{i}") for i, layer in enumerate(stage_vit_indices)])
            self.backbone_vit = IntermediateLayerGetter(backbone_vit, return_layers=return_vitlayers)

        # ---------------------------
        # 3. 构建 融合模块 (Fusion)
        # ---------------------------
        # 我们使用 ModuleList 来管理5个阶段的融合，方便在 forward 中循环调用
        self.fusion_layers = nn.ModuleList()

        for i in range(5):
            if not self.use_vit:
                # Row 1: 仅 CNN，无融合操作，占位符
                self.fusion_layers.append(nn.Identity())
            else:
                # 存在 ViT 分支
                if self.use_dcaf:
                    # Row 3 & 4: 使用 dcaf 注意力融合
                    # 假设 ChannelAttention 内部处理维度对齐或输入维度为 ViT 维度
                    self.fusion_layers.append(DCAF(in_channels=self.stage_vit_out_channels[i]))
                else:
                    # Row 2: 简单相加 (Simple Add)
                    # 如果 ViT 和 CNN 通道数不一致，需要 1x1 卷积将 ViT 对齐到 CNN
                    if self.stage_vit_out_channels[i] != self.stage_cnn_out_channels[i]:
                        self.fusion_layers.append(
                            ConvAlign(self.stage_vit_out_channels[i], self.stage_cnn_out_channels[i]))
                    else:
                        self.fusion_layers.append(nn.Identity())

        # ---------------------------
        # 4. 构建 解码器 (Decoder / UpSampling)
        # ---------------------------
        # 注意：无论是否融合，进入 Up 模块的通道数应当保持为 CNN 的通道数设计
        c = self.stage_cnn_out_channels[4] + self.stage_cnn_out_channels[3]
        self.up1 = Up(c, self.stage_cnn_out_channels[3])

        c = self.stage_cnn_out_channels[3] + self.stage_cnn_out_channels[2]
        self.up2 = Up(c, self.stage_cnn_out_channels[2])

        c = self.stage_cnn_out_channels[2] + self.stage_cnn_out_channels[1]
        self.up3 = Up(c, self.stage_cnn_out_channels[1])

        c = self.stage_cnn_out_channels[1] + self.stage_cnn_out_channels[0]
        self.up4 = Up(c, self.stage_cnn_out_channels[0])

        # ---------------------------
        # 5. 可选组件 (MLSP)
        # ---------------------------
        if self.use_mlsp:
            self.mlsp = MLSP(self.stage_cnn_out_channels[4], self.stage_cnn_out_channels[4])

        self.conv = OutConv(self.stage_cnn_out_channels[0], num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_shape = x.shape[-2:]

        # 1. 提取 CNN 特征
        backbone_cnnout = self.backbone_cnn(x)

        # 2. 提取 ViT 特征 (如果有)
        backbone_vitout = None
        if self.use_vit:
            backbone_vitout = self.backbone_vit(x)

        # 3. 特征融合循环
        features = []
        for i in range(5):
            stage_name = f'stage{i}'
            f_cnn = backbone_cnnout[stage_name]

            if not self.use_vit:
                # Row 1: 仅使用 CNN 特征
                x_fused = f_cnn
            else:
                f_vit = backbone_vitout[stage_name]

                if self.use_dcaf:
                    # Row 3 & 4: dcaf 融合 (调用 ChannelAttention)
                    # 注意：根据你之前的代码，ChannelAttention 接收 (cnn, vit)
                    x_fused = self.fusion_layers[i](f_cnn, f_vit)
                else:
                    # Row 2: 简单相加 (Element-wise Add)
                    # 先通过 fusion_layer (可能是 ConvAlign 或 Identity) 处理 ViT 特征以对齐通道
                    f_vit_aligned = self.fusion_layers[i](f_vit)
                    x_fused = f_cnn + f_vit_aligned

            features.append(x_fused)

        # 解包融合后的特征，命名习惯保留 x0-x4
        x0, x1, x2, x3, x4 = features[0], features[1], features[2], features[3], features[4]
        # 5. MLSP (如果有)
        if self.use_mlsp:
            x4 = self.mlsp(x4)
        # 4. 解码上采样
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

        # 6. 输出头
        x = self.conv(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        return x

if __name__ == '__main__':
    # 假设输入图像大小为 3x224x224，num_classes 根据需要调整
    model = PLMoViTUnet_small(num_classes=2,use_mlsp=True)

    # 打印模型结构
    print(model)

    # 使用 torchinfo 打印模型的详细摘要
    summary(model, input_size=(1, 3, 224, 224))  # 输入大小为 1 张 3x224x224 的图像