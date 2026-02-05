import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from datasetrp import CustomImageDataset
# from src.unet import UNet
from src.plmovitunet.PL_MoViT_Unet import PLMoViTUnet_large
from utils.losses1 import CombinedLoss
from utils.metrics1 import calculate_metrics
from utils.early_stopping import EarlyStopping
from PIL import Image
from tqdm import tqdm
import os
import random
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter


def set_seed(seed):
    """设置所有随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                writer, num_epochs, save_dir='./models', use_early_stopping=True,
                patience=15, min_delta=1e-4, calculate_hd95=False):
    """
    改进的训练函数,支持早停机制和完整指标记录

    Args:
        model: 训练模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        writer: TensorBoard writer
        num_epochs: 最大训练轮数
        save_dir: 模型保存目录
        use_early_stopping: 是否使用早停
        patience: 早停容忍轮数
        min_delta: 早停最小改善幅度
        calculate_hd95: 是否计算HD95(耗时较长,默认关闭)
    """
    best_dice = 0.0

    # 创建模型保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 初始化早停机制
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, mode='max')

    # 判断调度器类型
    is_onecycle = isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)

    for epoch in range(num_epochs):
        # ========== 训练阶段 ==========
        model.train()
        running_loss = 0.0
        running_dice = 0.0  # 肺实质 Dice
        running_iou = 0.0
        running_precision = 0.0
        running_recall = 0.0

        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as t:
            for batch_idx, (inputs, labels) in enumerate(t):
                inputs, labels = inputs.to(device), labels.to(device)

                # 清零梯度
                optimizer.zero_grad()

                # 前向传播
                outputs = model(inputs)

                # 计算损失
                loss = criterion(outputs, labels)

                # 反向传播
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # 优化器更新
                optimizer.step()

                # OneCycleLR 每个 batch 更新
                if is_onecycle:
                    scheduler.step()

                # 计算所有指标
                with torch.no_grad():
                    pred = torch.argmax(outputs, dim=1)
                    dice, iou, precision, recall = calculate_metrics(
                        pred, labels, n_classes=2, calculate_hd95=False
                    )

                    # 二分类任务:只取肺实质(类别1)的指标
                    running_dice += dice[0]
                    running_iou += iou[0]
                    running_precision += precision[0]
                    running_recall += recall[0]

                running_loss += loss.item()

                # 更新进度条
                t.set_postfix(
                    loss=running_loss / (t.n + 1),
                    dice=running_dice / (t.n + 1),
                    lr=optimizer.param_groups[0]['lr']
                )

        # 计算训练集平均指标
        num_batches = len(train_loader)
        epoch_loss = running_loss / num_batches
        epoch_metrics = {
            'dice': running_dice / num_batches,
            'iou': running_iou / num_batches,
            'precision': running_precision / num_batches,
            'recall': running_recall / num_batches,
        }

        # ========== 验证阶段 ==========
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        val_precision = 0.0
        val_recall = 0.0
        val_hd95 = 0.0
        hd95_count = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                pred = torch.argmax(outputs, dim=1)

                # 计算完整指标
                if calculate_hd95:
                    dice, iou, precision, recall, hd95 = calculate_metrics(
                        pred, labels, n_classes=2, calculate_hd95=True
                    )
                    if hd95[0] is not None:
                        val_hd95 += hd95[0]
                        hd95_count += 1
                else:
                    dice, iou, precision, recall = calculate_metrics(
                        pred, labels, n_classes=2, calculate_hd95=False
                    )

                val_dice += dice[0]
                val_iou += iou[0]
                val_precision += precision[0]
                val_recall += recall[0]
                val_loss += loss.item()

        # 计算验证集平均指标
        num_val_batches = len(val_loader)
        val_loss = val_loss / num_val_batches
        val_metrics = {
            'dice': val_dice / num_val_batches,
            'iou': val_iou / num_val_batches,
            'precision': val_precision / num_val_batches,
            'recall': val_recall / num_val_batches,
        }

        if calculate_hd95:
            val_metrics['hd95'] = val_hd95 / hd95_count if hd95_count > 0 else 0

        # ========== 打印结果 ==========
        tqdm.write(f'\n{"=" * 80}')
        tqdm.write(f'Epoch [{epoch + 1}/{num_epochs}]')
        tqdm.write(f'{"-" * 80}')
        tqdm.write(f'{"TRAIN":<10} | Loss: {epoch_loss:.4f}')
        tqdm.write(f'{"Lung":<10} | Dice: {epoch_metrics["dice"]:.4f} | IoU: {epoch_metrics["iou"]:.4f} | '
                   f'Prec: {epoch_metrics["precision"]:.4f} | Rec: {epoch_metrics["recall"]:.4f}')
        tqdm.write(f'{"-" * 80}')
        tqdm.write(f'{"VALID":<10} | Loss: {val_loss:.4f}')
        tqdm.write(f'{"Lung":<10} | Dice: {val_metrics["dice"]:.4f} | IoU: {val_metrics["iou"]:.4f} | '
                   f'Prec: {val_metrics["precision"]:.4f} | Rec: {val_metrics["recall"]:.4f}', end='')
        if calculate_hd95 and hd95_count > 0:
            tqdm.write(f' | HD95: {val_metrics["hd95"]:.2f}')
        else:
            tqdm.write('')

        tqdm.write(f'{"-" * 80}')
        tqdm.write(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        # ========== TensorBoard 记录 ==========
        # 训练指标
        writer.add_scalar('Loss/train', epoch_loss, epoch + 1)
        writer.add_scalar('Dice/train', epoch_metrics['dice'], epoch + 1)
        writer.add_scalar('IoU/train', epoch_metrics['iou'], epoch + 1)
        writer.add_scalar('Precision/train', epoch_metrics['precision'], epoch + 1)
        writer.add_scalar('Recall/train', epoch_metrics['recall'], epoch + 1)

        # 验证指标
        writer.add_scalar('Loss/valid', val_loss, epoch + 1)
        writer.add_scalar('Dice/valid', val_metrics['dice'], epoch + 1)
        writer.add_scalar('IoU/valid', val_metrics['iou'], epoch + 1)
        writer.add_scalar('Precision/valid', val_metrics['precision'], epoch + 1)
        writer.add_scalar('Recall/valid', val_metrics['recall'], epoch + 1)

        if calculate_hd95 and hd95_count > 0:
            writer.add_scalar('HD95/valid', val_metrics['hd95'], epoch + 1)

        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch + 1)

        # ========== 保存最优模型 ==========
        current_dice = val_metrics['dice']
        if current_dice > best_dice:
            best_dice = current_dice
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
                'val_metrics': val_metrics,
            }, os.path.join(save_dir, 'best_lung_PLMoViTUnet_large.pth'))
            tqdm.write(f'✓ Best model saved! Lung Dice: {best_dice:.4f}')

        # ========== 学习率调度 ==========
        if not is_onecycle:
            scheduler.step()

        # ========== 早停检查 ==========
        if use_early_stopping:
            if early_stopping(current_dice):
                tqdm.write(f'\n{"=" * 80}')
                tqdm.write(f'Early stopping triggered after {epoch + 1} epochs')
                tqdm.write(f'Best Dice: {best_dice:.4f}')
                tqdm.write(f'{"=" * 80}\n')
                break

            if early_stopping.counter > 0:
                tqdm.write(f'⚠ No improvement for {early_stopping.counter}/{patience} epochs')

        tqdm.write(f'{"=" * 80}\n')
        time.sleep(0.3)

    # 训练结束
    tqdm.write(f'\n{"=" * 80}')
    tqdm.write(f'Training completed!')
    tqdm.write(f'Best validation Dice: {best_dice:.4f}')
    tqdm.write(f'{"=" * 80}\n')


if __name__ == '__main__':
    # ========== 设备和随机种子 ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} to train...')

    seed = 3407
    set_seed(seed)
    print(f'Random seed is {seed}\n')

    # ========== 数据预处理 ==========
    # 肺实质分割:适度数据增强,避免大角度旋转
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15, interpolation=Image.NEAREST),  # 限制旋转角度
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
    ])

    # ========== 数据加载 ==========
    train_dataset = CustomImageDataset(data_type='train', transform=transform)
    val_dataset = CustomImageDataset(data_type='valid', transform=valid_transform)

    print(f'Dataset sizes:')
    print(f'  Train: {len(train_dataset)}')
    print(f'  Valid: {len(val_dataset)}\n')

    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f'DataLoader batches:')
    print(f'  Train: {len(train_loader)}')
    print(f'  Valid: {len(val_loader)}\n')

    # ========== TensorBoard ==========
    writer = SummaryWriter('runsjn/PLMoViTUnet_large_Lung_Seg')

    # ========== 模型 ==========
    # 肺实质分割:二分类(背景+肺实质)
    model = PLMoViTUnet_large(num_classes=2)
    model = model.to(device)
    print(f'Model loaded: PLMoViTUnet_large (Binary Segmentation)\n')

    # ========== 损失函数 ==========
    criterion = CombinedLoss()

    # ========== 优化器和学习率调度器 ==========
    num_epochs = 100

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4
    )
    print('Optimizer: AdamW')
    print('Scheduler: OneCycleLR with warmup\n')

    # ========== 训练参数 ==========
    print('Training configuration:')
    print(f'  Task: Lung Parenchyma Segmentation (Binary)')
    print(f'  Model: PLMoViTUnet_large')
    print(f'  Epochs: {num_epochs}')
    print(f'  Batch size: {batch_size}')
    print(f'  Early stopping: Enabled (patience=15)')
    print(f'  Gradient clipping: Enabled (max_norm=1.0)')
    print(f'  HD95 calculation: Disabled (set calculate_hd95=True to enable)')
    print(f'{"=" * 80}\n')

    # ========== 开始训练 ==========
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        writer=writer,
        num_epochs=num_epochs,
        save_dir='./modelsjn',
        use_early_stopping=True,
        patience=20,
        min_delta=1e-4,
        calculate_hd95=False
    )

    writer.close()
