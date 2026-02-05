import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

# 导入你的模块
from dataset import CustomImageDataset
from src.plmovitunet.PL_MoViT_Unet import PLMoViTUnet_large


# ================= 1. DCAF 权重追踪器 (核心类) =================
class DCAFWeightTracker:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.weight_history = {}  # {stage_idx: [cnn_weights]}
        self._register_hooks()

    def _register_hooks(self):
        idx = 0
        for name, module in self.model.named_modules():
            # 自动识别所有 DCAF 模块中的权重生成器
            if hasattr(module, 'weight_generator'):
                self.weight_history[idx] = []
                handle = module.weight_generator.register_forward_hook(self._get_hook(idx))
                self.handles.append(handle)
                idx += 1
        print(f"🎯 系统已识别并追踪到 {idx} 个 DCAF 融合层。")

    def _get_hook(self, idx):
        def hook_fn(module, input, output):
            # output: [B, 2, 1, 1] -> Softmax 后的权重
            # 取通道0作为 CNN 权重 (Alpha)
            cnn_w = output[:, 0, 0, 0].detach().cpu().numpy()
            self.weight_history[idx].extend(cnn_w.tolist())

        return hook_fn

    def remove(self):
        for h in self.handles: h.remove()

    def plot_and_save(self, save_path):
        num_stages = len(self.weight_history)
        if num_stages == 0: return

        fig, axes = plt.subplots(1, num_stages, figsize=(5 * num_stages, 5))
        if num_stages == 1: axes = [axes]

        print("\n📊 正在生成分布统计图...")
        for i in range(num_stages):
            data = np.array(self.weight_history[i])
            axes[i].hist(data, bins=40, color='#4e79a7', edgecolor='white', alpha=0.8)

            avg = np.mean(data)
            std = np.std(data)

            axes[i].axvline(avg, color='#e15759', linestyle='--', linewidth=2, label=f'Mean: {avg:.3f}')
            axes[i].set_title(f'Stage {i} (Bottleneck)\nCNN vs Trans', fontsize=12)
            axes[i].set_xlabel('CNN Weight ($\mu$)', fontsize=10)
            axes[i].set_xlim(0, 1)
            axes[i].legend()

            # 打印控制台摘要
            print(f"Stage {i} -> 平均 CNN 权重: {avg:.4f} | 标准差: {std:.4f}")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ 权重统计分布图已保存至: {save_path}")


def run_weight_analysis(model_path, data_root, batch_size=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 当前设备: {device}")

    # 1. 加载模型
    model = PLMoViTUnet_large(num_classes=3).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()

    # 2. 初始化追踪器
    tracker = DCAFWeightTracker(model)

    # 3. 准备数据 - 使用简化的 transform
    # 假设 dataset 已经返回张量
    test_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        # 不要添加 ToTensor()，dataset 可能已经处理了
    ])
    
    # 或者如果 dataset 返回 PIL 图像，则添加 ToTensor()
    # test_transform = transforms.Compose([
    #     transforms.Resize((256, 256), interpolation=Image.NEAREST),
    #     transforms.ToTensor(),  # 根据实际情况决定是否添加
    # ])
    
    test_dataset = CustomImageDataset(data_type='test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 4. 尝试推理，如果有错误则调整
    print("🚀 开始遍历数据集以提取 DCAF 动态权重...")
    with torch.no_grad():
        try:
            for inputs, _ in tqdm(test_loader):
                _ = model(inputs.to(device))
        except Exception as e:
            print(f"❌ 推理出错: {e}")
            print("尝试调整 transform...")
            
            # 尝试不同的 transform
            test_transform = transforms.Compose([
                transforms.Resize((256, 256), interpolation=Image.NEAREST),
            ])
            
            test_dataset = CustomImageDataset(data_type='test', transform=test_transform)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            
            for inputs, _ in tqdm(test_loader):
                _ = model(inputs.to(device))

    # 5. 生成结果
    os.makedirs('./analysis_results', exist_ok=True)
    tracker.plot_and_save('./analysis_results/dcaf_weight_report.png')
    tracker.remove()


if __name__ == '__main__':
    # 修改为你实际的路径
    MY_MODEL_PATH = './models/best_PLMoViTUnet_large.pth'

    if os.path.exists(MY_MODEL_PATH):
        run_weight_analysis(MY_MODEL_PATH, data_root='./data')
    else:
        print(f"⚠️ 找不到权重文件: {MY_MODEL_PATH}")