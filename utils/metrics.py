import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff

#utils/metrics1
# 计算指标函数
def calculate_metrics(preds, targets, n_classes=3, calculate_hd95=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dice_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []
    hd95_scores = []

    for cls in range(1, n_classes):  # 忽略背景
        num = targets.shape[0]
        dice_sum = torch.tensor([0.0]).to(device)
        iou_sum = torch.tensor([0.0]).to(device)
        precision_sum = torch.tensor([0.0]).to(device)
        recall_sum = torch.tensor([0.0]).to(device)
        hd95_sum = 0.0
        hd95_count = 0

        for i in range(0, num):  # 对batch中的每个实例进行计算
            pred = preds[i]
            target = targets[i]
            pred_cls = (pred == cls)
            target_cls = (target == cls)

            # 基础计算
            intersection = torch.logical_and(pred_cls, target_cls).sum().float()
            union = torch.logical_or(pred_cls, target_cls).sum().float()
            pred_sum = pred_cls.sum().float()
            target_sum = target_cls.sum().float()

            # Dice 和 IoU
            if intersection + union > 0:
                dice = (2. * intersection) / (pred_sum + target_sum + 1e-6)
                iou = intersection / (union + 1e-6)
            else:
                dice = torch.tensor([1.0]).to(device)
                iou = torch.tensor([1.0]).to(device)

            # Precision 和 Recall
            if pred_sum > 0:
                precision = intersection / (pred_sum + 1e-6)
            else:
                precision = torch.tensor([1.0]).to(device) if target_sum == 0 else torch.tensor([0.0]).to(device)

            if target_sum > 0:
                recall = intersection / (target_sum + 1e-6)
            else:
                recall = torch.tensor([1.0]).to(device) if pred_sum == 0 else torch.tensor([0.0]).to(device)

            dice_sum += dice
            iou_sum += iou
            precision_sum += precision
            recall_sum += recall

            # HD95 计算（仅在有前景像素时计算）
            if calculate_hd95 and pred_sum > 0 and target_sum > 0:
                hd95 = compute_hd95(pred_cls.cpu().numpy(), target_cls.cpu().numpy())
                if hd95 is not None:
                    hd95_sum += hd95
                    hd95_count += 1

        # 计算平均值
        dice_per = dice_sum / num
        iou_per = iou_sum / num
        precision_per = precision_sum / num
        recall_per = recall_sum / num

        dice_scores.append(dice_per.item())
        iou_scores.append(iou_per.item())
        precision_scores.append(precision_per.item())
        recall_scores.append(recall_per.item())

        if calculate_hd95:
            hd95_per = hd95_sum / hd95_count if hd95_count > 0 else None
            hd95_scores.append(hd95_per)

    if calculate_hd95:
        return dice_scores, iou_scores, precision_scores, recall_scores, hd95_scores
    else:
        return dice_scores, iou_scores, precision_scores, recall_scores


def compute_hd95(pred, target):
    """
    计算95%的Hausdorff距离
    Args:
        pred: 二值预测mask (numpy array)
        target: 二值真值mask (numpy array)
    Returns:
        hd95: 95百分位的Hausdorff距离
    """
    # 提取边界点
    pred_points = np.argwhere(pred)
    target_points = np.argwhere(target)

    if len(pred_points) == 0 or len(target_points) == 0:
        return None

    # 计算双向Hausdorff距离
    hd_pred_to_target = directed_hausdorff(pred_points, target_points)[0]
    hd_target_to_pred = directed_hausdorff(target_points, pred_points)[0]

    # 计算所有点到最近点的距离
    from scipy.spatial.distance import cdist
    distances_1 = cdist(pred_points, target_points).min(axis=1)
    distances_2 = cdist(target_points, pred_points).min(axis=1)
    all_distances = np.concatenate([distances_1, distances_2])

    # 返回95百分位数
    hd95 = np.percentile(all_distances, 95)
    return hd95


if __name__ == '__main__':
    pred = torch.rand(2, 3, 4, 4)
    pred = torch.argmax(pred, dim=1)
    target = torch.tensor(
        [[[1, 2, 0, 0],
          [0, 0, 1, 1],
          [0, 2, 0, 0],
          [2, 1, 0, 0]],
         [[1, 2, 0, 0],
          [0, 0, 1, 1],
          [0, 2, 0, 0],
          [2, 1, 0, 0]]
         ]
    )

    # 测试（不计算HD95，因为样本太小）
    dice, iou, precision, recall = calculate_metrics(pred, target, calculate_hd95=False)

    print(f"Dice: {dice}")
    print(f"IoU: {iou}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    # 如需计算HD95（需要安装scipy）
    # dice, iou, precision, recall, hd95 = calculate_metrics(pred, target, calculate_hd95=True)
    # print(f"HD95: {hd95}")
