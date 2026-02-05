import csv
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from datasetrp import CustomImageDataset  # ⭐ 修改:使用肺实质数据集
from src.plmovitunet.PL_MoViT_Unet import PLMoViTUnet_large
from tqdm import tqdm
import os
import random
import numpy as np
from PIL import Image

from utils.losses1 import CombinedLoss
from utils.metrics1 import calculate_metrics

# ⭐ 修改:二分类标签映射 (背景 + 肺实质)
label_mapping = {
    0: 0,    # 背景 -> 类别0
    255: 1   # 肺实质 -> 类别1
}
# 逆向映射（用于预测结果）
inverse_mapping = {v: k for k, v in label_mapping.items()}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_image(images, preds, labels, name, save_dir='./results/img_pred_gt'):
    """保存图像、预测和真值的可视化对比"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    images = images.to('cpu')
    preds = preds.to('cpu')
    labels = labels.to('cpu')

    images = (images * 255).to(torch.uint8)  # [0,1] to [0,255]
    preds = torch.from_numpy(np.vectorize(inverse_mapping.get)(preds.numpy()))
    preds = torch.stack((preds, preds, preds), dim=1).to(torch.uint8)
    labels = torch.from_numpy(np.vectorize(inverse_mapping.get)(labels.numpy()))
    labels = torch.stack((labels, labels, labels), dim=1).to(torch.uint8)

    grid_img = make_grid(images)
    grid_pred = make_grid(preds)
    grid_label = make_grid(labels)
    concat = torch.cat((grid_img, grid_pred, grid_label), dim=1)
    concat = concat.permute(2, 1, 0)
    concat = Image.fromarray(concat.numpy())
    concat.save(os.path.join(save_dir, name))


def test_model(model, test_loader, criterion, save_dir='./resultsjn', calculate_hd95=True, save_images=True):
    """
    完整的测试函数,支持所有评估指标 (二分类版本)

    Args:
        model: 测试模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        save_dir: 结果保存目录
        calculate_hd95: 是否计算HD95(耗时较长)
        save_images: 是否保存可视化图像
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()

    # ⭐ 修改:初始化单类别指标列表
    test_loss = 0.0
    test_dice = []      # 肺实质 Dice
    test_iou = []
    test_precision = []
    test_recall = []
    test_hd95 = []

    image_count = 0
    max_images = 1000  # 最多保存1000张图像

    print(f'\n{"=" * 80}')
    print(f'Starting model evaluation...')
    print(f'Task: Lung Parenchyma Segmentation (Binary)')
    print(f'Calculate HD95: {calculate_hd95}')
    print(f'Save visualizations: {save_images} (max {max_images} images)')
    print(f'{"=" * 80}\n')

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            # ⭐ 修改:计算二分类指标
            if calculate_hd95:
                dice, iou, precision, recall, hd95 = calculate_metrics(
                    preds, labels, n_classes=2, calculate_hd95=True
                )
                # HD95可能为None(当mask为空时)
                if hd95[0] is not None:
                    test_hd95.append(hd95[0])
            else:
                dice, iou, precision, recall = calculate_metrics(
                    preds, labels, n_classes=2, calculate_hd95=False
                )

            # 记录指标 (只记录肺实质,即类别1)
            test_dice.append(dice[0])
            test_iou.append(iou[0])
            test_precision.append(precision[0])
            test_recall.append(recall[0])

            # 保存可视化图像
            if save_images and image_count < max_images:
                name = f'{image_count:04d}.png'
                save_image(inputs, preds, labels, name, save_dir=os.path.join(save_dir, 'visualizations'))
                image_count += 1

            # 计算损失
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    # 计算平均值
    test_loss = test_loss / len(test_loader)

    metrics_summary = {
        'Dice': np.mean(test_dice),
        'IoU': np.mean(test_iou),
        'Precision': np.mean(test_precision),
        'Recall': np.mean(test_recall),
    }

    if calculate_hd95 and test_hd95:
        metrics_summary['HD95'] = np.mean(test_hd95)

    # ========== 打印结果 ==========
    print(f'\n{"=" * 80}')
    print(f'TEST RESULTS - LUNG PARENCHYMA SEGMENTATION')
    print(f'{"=" * 80}')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'{"-" * 80}')
    print(f'{"Metric":<15} | {"Value":<12}')
    print(f'{"-" * 80}')

    for metric_name, value in metrics_summary.items():
        if metric_name == 'HD95':
            print(f'{metric_name:<15} | {value:<12.2f}')
        else:
            print(f'{metric_name:<15} | {value:<12.4f}')

    print(f'{"=" * 80}\n')

    # ========== 保存详细结果到CSV ==========
    csv_path = os.path.join(save_dir, 'detailed_metrics.csv')

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # 写入列标题
        headers = ['Sample_ID', 'Dice', 'IoU', 'Precision', 'Recall']
        if calculate_hd95:
            headers.append('HD95')
        writer.writerow(headers)

        # 写入每个样本的指标
        for i in range(len(test_dice)):
            row = [
                i,
                test_dice[i],
                test_iou[i],
                test_precision[i],
                test_recall[i],
            ]
            if calculate_hd95:
                row.append(test_hd95[i] if i < len(test_hd95) else 'N/A')
            writer.writerow(row)

        # 写入平均值
        avg_row = ['Average']
        for metric_name in ['Dice', 'IoU', 'Precision', 'Recall']:
            avg_row.append(f'{metrics_summary[metric_name]:.4f}')
        if calculate_hd95 and 'HD95' in metrics_summary:
            avg_row.append(f'{metrics_summary["HD95"]:.2f}')
        writer.writerow(avg_row)

    print(f'✓ Detailed metrics saved to: {csv_path}')

    # ========== 保存汇总统计到CSV ==========
    summary_path = os.path.join(save_dir, 'summary_statistics.csv')

    with open(summary_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Value'])

        for metric_name, value in metrics_summary.items():
            if metric_name == 'HD95':
                writer.writerow([metric_name, f'{value:.2f}'])
            else:
                writer.writerow([metric_name, f'{value:.4f}'])

        writer.writerow(['Loss', f'{test_loss:.4f}'])

    print(f'✓ Summary statistics saved to: {summary_path}')

    if save_images:
        print(f'✓ {image_count} visualization images saved to: {os.path.join(save_dir, "visualizations")}')

    print(f'\n{"=" * 80}\n')

    return metrics_summary


if __name__ == '__main__':
    # ========== 设备和随机种子 ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} to test...')

    seed = 3407
    set_seed(seed)
    print(f'Random seed is {seed}\n')

    # ========== 数据预处理 ==========
    test_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
    ])

    # ========== 数据加载 ==========
    test_dataset = CustomImageDataset(data_type='test', transform=test_transform)
    print(f'Test dataset size: {len(test_dataset)}')

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    print(f'Test batches: {len(test_loader)}\n')

    # ========== 模型加载 ==========
    print(f'Loading model from ./modelsjn/best_lung_PLMoViTUnet_large.pth...')
    model = PLMoViTUnet_large(num_classes=2)  # ⭐ 修改:二分类
    model = model.to(device)

    checkpoint = torch.load('./modelsjn/best_lung_PLMoViTUnet_large.pth', map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'✓ Model loaded (epoch {checkpoint.get("epoch", "N/A")})')
        if 'best_dice' in checkpoint:
            print(f'  Best validation Dice: {checkpoint["best_dice"]:.4f}')
    else:
        model.load_state_dict(checkpoint)
        print(f'✓ Model loaded')

    # ========== 损失函数 ==========
    criterion = CombinedLoss()

    # ========== 测试配置 ==========
    print(f'\nTest configuration:')
    print(f'  Task: Lung Parenchyma Segmentation')
    print(f'  Model:PLMoViTUnet_large (Binary)')
    print(f'  Test samples: {len(test_dataset)}')
    print(f'  Calculate HD95: True')
    print(f'  Save visualizations: True')
    print(f'{"=" * 80}\n')

    # ========== 开始测试 ==========
    metrics_summary = test_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        save_dir='./resultsjn/PLMoViTUnet_large_Lung_Test',  # ⭐ 修改保存目录
        calculate_hd95=True,  # 改为False可跳过HD95计算以加快速度
        save_images=True      # 是否保存可视化图像
    )
