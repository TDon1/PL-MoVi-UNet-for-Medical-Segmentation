import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from datasetrp import CustomImageDataset  # ⭐ 修改:使用肺实质数据集
from src.unet import UNet
from src.unet_pp import UNetPP
from src.attention_unet import AttU_Net
from src.setr.SETR import SETR_Naive_S
from src.deeplabv3_model import deeplabv3_resnet50
from src.transunet.vit_seg_modeling import VisionTransformer, CONFIGS
from src.plmovitunet.PL_MoViT_Unet import PLMoViTUnet_large,PLMoViTUnet_small
def create_model_unetpp(in_channel, num_classes, deep_supervision=False):
    """
    创建 UNet++ 模型

    Args:
        in_channel: 输入通道数 (RGB图像为3)
        num_classes: 分类数量
        deep_supervision: 是否使用深度监督
    """
    model = UNetPP(
        in_channel=in_channel,
        out_channel=num_classes,
        features=[64, 128, 256, 512, 1024],
        deep_supervision=deep_supervision
    )
    return model

def create_model_attunet(in_channel, num_classes):
    """
    创建 Attention U-Net 模型

    Args:
        in_channel: 输入通道数 (RGB图像为3)
        num_classes: 分类数量
    """
    model = AttU_Net(img_ch=in_channel, output_ch=num_classes)
    return model

def create_model_deeplabv3(aux, num_classes):
    model = deeplabv3_resnet50(aux=aux, num_classes=num_classes)
    return model
def create_setr_model(num_classes):
    #model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    #model = VGG16UNet(num_classes=num_classes, pretrain_backbone=False)
    aux_layers, model = SETR_Naive_S(dataset='pascal', _conv_repr=False, _pe_type="learned")
    return model

from src.swin_unet import SwinUnet  # ⭐ 修改:导入 Swin-UNet
import ml_collections
def get_swin_unet_config():
    """
    创建 Swin-UNet 的配置
    """
    config = ml_collections.ConfigDict()

    # 数据配置
    config.DATA = ml_collections.ConfigDict()
    config.DATA.IMG_SIZE = 256  # 图像大小

    # 模型配置
    config.MODEL = ml_collections.ConfigDict()
    config.MODEL.DROP_RATE = 0.0
    config.MODEL.DROP_PATH_RATE = 0.1
    config.MODEL.PRETRAIN_CKPT = None  # 预训练权重路径,如果有的话

    # Swin Transformer 配置
    config.MODEL.SWIN = ml_collections.ConfigDict()
    config.MODEL.SWIN.PATCH_SIZE = 4
    config.MODEL.SWIN.IN_CHANS = 3
    config.MODEL.SWIN.EMBED_DIM = 96
    config.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    config.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    config.MODEL.SWIN.WINDOW_SIZE = 8
    config.MODEL.SWIN.MLP_RATIO = 4.0
    config.MODEL.SWIN.QKV_BIAS = True
    config.MODEL.SWIN.QK_SCALE = None
    config.MODEL.SWIN.APE = False
    config.MODEL.SWIN.PATCH_NORM = True

    # 训练配置
    config.TRAIN = ml_collections.ConfigDict()
    config.TRAIN.USE_CHECKPOINT = False

    return config
def create_model_swinunet(num_classes, img_size=256):
    """
    创建 Swin-UNet 模型

    Args:
        num_classes: 分类数量
        img_size: 输入图像大小
    """
    config = get_swin_unet_config()
    config.DATA.IMG_SIZE = img_size

    model = SwinUnet(
        config=config,
        img_size=img_size,
        num_classes=num_classes
    )

    # 如果有预训练权重,加载它
    model.load_from(config)

    return model

def create_model_transunet(num_classes):
    img_size =256

    vit_patches_size = 16
    vit_name='R50-ViT-B_16'
    config_vit = CONFIGS[vit_name] # 选择 ViT-B_16 配置
    config_vit.n_classes = num_classes
    config_vit.n_skip = 3
  # 根据需要设置跳跃连接的数量
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    model=VisionTransformer(config_vit, img_size=img_size, num_classes=num_classes)

    return model
import torch
from thop import profile, clever_format


def get_model_profile(model, input_size=(1, 3, 256, 256), device='cuda'):
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(input_size).to(device)

    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    return params, flops
def benchmark_all_models(num_classes=2, device='cuda'):
    print("{:<22} {:<12} {:<12}".format("Model", "Params(M)", "FLOPs(G)"))
    print("-" * 50)

    models = {
        "UNet++": create_model_unetpp(3, num_classes, False),
        "Attention U-Net": create_model_attunet(3, num_classes),
        "DeepLabV3": create_model_deeplabv3(False, num_classes),
        "SETR": create_setr_model(num_classes),
        "Swin-Unet": create_model_swinunet(num_classes),
        "TransUNet": create_model_transunet(num_classes),
        "PL-MoViT-Unet-S": PLMoViTUnet_small(num_classes=num_classes),
        "PL-MoViT-Unet-L": PLMoViTUnet_large(num_classes=num_classes),
    }

    for name, model in models.items():
        try:
            params, flops = get_model_profile(model)
            print("{:<22} {:<12} {:<12}".format(name, params, flops))
        except Exception as e:
            print("{:<22} {:<12} {:<12}".format(name, "Error", "Error"))
            print(f"  ↳ {e}")

if __name__ == "__main__":
    benchmark_all_models(num_classes=2, device='cuda')

