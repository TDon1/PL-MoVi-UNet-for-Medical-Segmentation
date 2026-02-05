import csv
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from dataset import CustomImageDataset
# from src.unet import UNet
from src.plmovitunet.PL_MoViT_Unet import PLMoViTUnet_large
from tqdm import tqdm
import os
import random
import numpy as np
from PIL import Image

from utils.losses1 import CombinedLoss
from utils.metrics1 import calculate_metrics

label_mapping = {
    0: 0,  # 背景 -> 类别0
    128: 1,  # 器官A -> 类别1
    255: 2  # 器官B -> 类别2
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


def test_model(model, test_loader, criterion, save_dir='./results', calculate_hd95=True, save_images=True):
    """
    完整的测试函数，支持所有评估指标

    Args:
        model: 测试模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        save_dir: 结果保存目录
        calculate_hd95: 是否计算HD95（耗时较长）
        save_images: 是否保存可视化图像
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()

    # 初始化所有指标列表
    test_loss = 0.0
    test_dice_1, test_dice_2 = [], []
    test_iou_1, test_iou_2 = [], []
    test_precision_1, test_precision_2 = [], []
    test_recall_1, test_recall_2 = [], []
    test_hd95_1, test_hd95_2 = [], []

    image_count = 0
    max_images = 1000  # 最多保存1000张图像

    print(f'\n{"=" * 80}')
    print(f'Starting model evaluation...')
    print(f'Calculate HD95: {calculate_hd95}')
    print(f'Save visualizations: {save_images} (max {max_images} images)')
    print(f'{"=" * 80}\n')

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            # 计算所有指标
            if calculate_hd95:
                dice, iou, precision, recall, hd95 = calculate_metrics(
                    preds, labels, n_classes=3, calculate_hd95=True
                )
                # HD95可能为None（当mask为空时）
                if hd95[0] is not None:
                    test_hd95_1.append(hd95[0])
                if hd95[1] is not None:
                    test_hd95_2.append(hd95[1])
            else:
                dice, iou, precision, recall = calculate_metrics(
                    preds, labels, n_classes=3, calculate_hd95=False
                )

            # 记录指标
            test_dice_1.append(dice[0])
            test_dice_2.append(dice[1])
            test_iou_1.append(iou[0])
            test_iou_2.append(iou[1])
            test_precision_1.append(precision[0])
            test_precision_2.append(precision[1])
            test_recall_1.append(recall[0])
            test_recall_2.append(recall[1])

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
        'Dice_Class1': np.mean(test_dice_1),
        'Dice_Class2': np.mean(test_dice_2),
        'IoU_Class1': np.mean(test_iou_1),
        'IoU_Class2': np.mean(test_iou_2),
        'Precision_Class1': np.mean(test_precision_1),
        'Precision_Class2': np.mean(test_precision_2),
        'Recall_Class1': np.mean(test_recall_1),
        'Recall_Class2': np.mean(test_recall_2),
    }

    if calculate_hd95:
        metrics_summary['HD95_Class1'] = np.mean(test_hd95_1) if test_hd95_1 else None
        metrics_summary['HD95_Class2'] = np.mean(test_hd95_2) if test_hd95_2 else None

    # ========== 打印结果 ==========
    print(f'\n{"=" * 80}')
    print(f'TEST RESULTS')
    print(f'{"=" * 80}')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'{"-" * 80}')
    print(f'{"Metric":<15} | {"Class 1":<12} | {"Class 2":<12} | {"Average":<12}')
    print(f'{"-" * 80}')

    for metric in ['Dice', 'IoU', 'Precision', 'Recall']:
        class1_val = metrics_summary[f'{metric}_Class1']
        class2_val = metrics_summary[f'{metric}_Class2']
        avg_val = (class1_val + class2_val) / 2
        print(f'{metric:<15} | {class1_val:<12.4f} | {class2_val:<12.4f} | {avg_val:<12.4f}')

    if calculate_hd95:
        hd95_1 = metrics_summary.get('HD95_Class1')
        hd95_2 = metrics_summary.get('HD95_Class2')
        if hd95_1 is not None and hd95_2 is not None:
            avg_hd95 = (hd95_1 + hd95_2) / 2
            print(f'{"HD95":<15} | {hd95_1:<12.2f} | {hd95_2:<12.2f} | {avg_hd95:<12.2f}')

    print(f'{"=" * 80}\n')

    # ========== 保存详细结果到CSV ==========
    csv_path = os.path.join(save_dir, 'detailed_metrics.csv')

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # 写入列标题
        headers = ['Sample_ID', 'Dice_Class1', 'Dice_Class2', 'IoU_Class1', 'IoU_Class2',
                   'Precision_Class1', 'Precision_Class2', 'Recall_Class1', 'Recall_Class2']
        if calculate_hd95:
            headers.extend(['HD95_Class1', 'HD95_Class2'])
        writer.writerow(headers)

        # 写入每个样本的指标
        for i in range(len(test_dice_1)):
            row = [
                i,
                test_dice_1[i],
                test_dice_2[i],
                test_iou_1[i],
                test_iou_2[i],
                test_precision_1[i],
                test_precision_2[i],
                test_recall_1[i],
                test_recall_2[i],
            ]
            if calculate_hd95:
                row.append(test_hd95_1[i] if i < len(test_hd95_1) else 'N/A')
                row.append(test_hd95_2[i] if i < len(test_hd95_2) else 'N/A')
            writer.writerow(row)

        # 写入平均值
        avg_row = ['Average'] + [f'{v:.4f}' if isinstance(v, float) else 'N/A'
                                 for v in metrics_summary.values()]
        writer.writerow(avg_row)

    print(f'✓ Detailed metrics saved to: {csv_path}')

    # ========== 保存汇总统计到CSV ==========
    summary_path = os.path.join(save_dir, 'summary_statistics.csv')

    with open(summary_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Class 1', 'Class 2', 'Average'])

        for metric in ['Dice', 'IoU', 'Precision', 'Recall']:
            class1_val = metrics_summary[f'{metric}_Class1']
            class2_val = metrics_summary[f'{metric}_Class2']
            avg_val = (class1_val + class2_val) / 2
            writer.writerow([metric, f'{class1_val:.4f}', f'{class2_val:.4f}', f'{avg_val:.4f}'])

        if calculate_hd95:
            hd95_1 = metrics_summary.get('HD95_Class1')
            hd95_2 = metrics_summary.get('HD95_Class2')
            if hd95_1 is not None and hd95_2 is not None:
                avg_hd95 = (hd95_1 + hd95_2) / 2
                writer.writerow(['HD95', f'{hd95_1:.2f}', f'{hd95_2:.2f}', f'{avg_hd95:.2f}'])

        writer.writerow(['Loss', '', '', f'{test_loss:.4f}'])

    print(f'✓ Summary statistics saved to: {summary_path}')

    if save_images:
        print(f'✓ {image_count} visualization images saved to: {os.path.join(save_dir, "visualizations")}')

    print(f'\n{"=" * 80}\n')

    return metrics_summary


if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} to test...')

    # 设置随机种子
    seed = 3407
    set_seed(seed)
    print(f'Random seed is {seed}\n')

    # 数据预处理
    test_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
    ])

    # 加载测试集
    test_dataset = CustomImageDataset(data_type='test', transform=test_transform)
    print(f'Test dataset size: {len(test_dataset)}')

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    print(f'Test batches: {len(test_loader)}\n')

    # 导入模型
    print(f'Loading model from ./models/best_PLMoViTUnet_large.pth...')
    model = PLMoViTUnet_large(num_classes=3)
    model = model.to(device)

    checkpoint = torch.load('./models/best_PLMoViTUnet_large.pth', map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'✓ Model loaded (epoch {checkpoint.get("epoch", "N/A")})')
    else:
        model.load_state_dict(checkpoint)
        print(f'✓ Model loaded')

    # 损失函数
    criterion = CombinedLoss()

    # 开始测试
    metrics_summary = test_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        save_dir='./results/PLMoViTUnet_large',
        calculate_hd95=True,  # 改为False可跳过HD95计算以加快速度
        save_images=True  # 是否保存可视化图像
    )
