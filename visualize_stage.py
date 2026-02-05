import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms

# 导入项目内部模块
try:
    from src.plmovitunet.PL_MoViT_Unet import PLMoViTUnet_large
    from dataset import CustomImageDataset
except ImportError:
    print("❌ 导入失败，请确保在项目根目录下运行此脚本。")

def visualize_stage4_batch(model_path, sample_indices=range(10)):
    """
    批量生成 Stage 4 的特征对比图
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 当前设备: {device}")
    
    # 1. 加载模型
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    saved_classes = state_dict['conv.0.bias'].shape[0]
    
    model = PLMoViTUnet_large(num_classes=saved_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"⚙️ 模型加载成功，类别数: {saved_classes}")
    
    # 2. 准备数据 (保持与训练一致的尺寸)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.NEAREST),
    ])
    
    try:
        dataset = CustomImageDataset(data_type='test', transform=test_transform)
        print(f"📚 数据集已就绪，准备处理样本索引: {list(sample_indices)}")
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return

    output_dir = './stage4_more_samples'
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for idx in sample_indices:
            try:
                # 获取数据
                image_tensor, _ = dataset[idx]
                input_batch = image_tensor.unsqueeze(0).to(device)

                # 3. 提取 Stage 4 特征
                cnn_outputs = model.backbone_cnn(input_batch)
                vit_outputs = model.backbone_vit(input_batch)
                
                f_cnn = cnn_outputs['stage4'] 
                f_vit = vit_outputs['stage4'] 

                # 4. 计算并归一化热力图
                def process_feat(feat):
                    h = torch.mean(feat, dim=1).squeeze().cpu().numpy()
                    return (h - h.min()) / (h.max() - h.min() + 1e-8)

                heatmap_cnn = process_feat(f_cnn)
                heatmap_vit = process_feat(f_vit)

                # 5. 绘图
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # 原图反归一化显示
                img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
                if img_np.min() < 0: img_np = (img_np * 0.5) + 0.5
                
                axes[0].imshow(np.clip(img_np, 0, 1))
                axes[0].set_title(f"Original Image (Idx {idx})", fontsize=12)
                axes[0].axis('off')

                # CNN 支路可视化
                im1 = axes[1].imshow(heatmap_cnn, cmap='jet')
                axes[1].set_title("Stage 4: CNN Branch\n(Low Importance)", fontsize=11)
                axes[1].axis('off')
                plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

                # ViT 支路可视化
                im2 = axes[2].imshow(heatmap_vit, cmap='jet')
                axes[2].set_title("Stage 4: ViT Branch\n(High Importance)", fontsize=11)
                axes[2].axis('off')
                plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

                plt.tight_layout()
                save_path = os.path.join(output_dir, f"stage4_sample_{idx}.png")
                plt.savefig(save_path, dpi=200)
                plt.close()
                print(f"✅ 样本 {idx} 处理完成 -> {save_path}")
                
            except Exception as e:
                print(f"⚠️ 处理样本 {idx} 时跳过: {e}")

if __name__ == '__main__':
    MY_MODEL_PATH = './models/best_PLMoViTUnet_large.pth'
    
    # 这里您可以自定义想要查看的索引范围
    # 示例：range(10) 表示前 10 张，或者 [15, 22, 30] 挑特定的看
    visualize_stage4_batch(MY_MODEL_PATH, sample_indices=range(50))