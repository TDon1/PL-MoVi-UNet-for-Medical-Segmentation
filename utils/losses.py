import torch
import torch.nn as nn
import torch.nn.functional as F

#utils/losses1
class TverskyLoss(nn.Module):
    def __init__(self,
                 alpha=0.3,  # FP 惩罚系数
                 beta=0.7,  # FN 惩罚系数 (beta > alpha 提高召回率)
                 gamma=1.0,  # Focal 系数 (1.0 即为普通 Tversky)
                 smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, target):
        """
        logits: [B, C, H, W] - 模型输出
        target: [B, H, W] - 真实标签
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # 1. 转换标签为 One-Hot [B, C, H, W]
        target_oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

        # 2. 计算各类的 TP, FP, FN [B, C]
        # 对 H, W 维度求和
        tp = (probs * target_oh).sum(dim=(2, 3))
        fp = (probs * (1 - target_oh)).sum(dim=(2, 3))
        fn = ((1 - probs) * target_oh).sum(dim=(2, 3))

        # 3. 计算 Tversky 指数
        # 加上 smooth 防止分母为 0
        tversky_idx = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # 4. Focal 变换并计算 Loss
        # 1 - tversky_idx 范围在 [0, 1] 之间
        loss = torch.pow((1 - tversky_idx), self.gamma)

        # 5. 通常忽略背景类 (Class 0) 或对所有类取平均
        # 在医学分割中，建议只取前景类 [:, 1:]
        return loss[:, 1:].mean() if num_classes > 1 else loss.mean()


class CombinedLoss(nn.Module):
    """
    推荐方案：CrossEntropy 保证全局梯度，Tversky 负责边界和召回率
    """

    def __init__(self, ce_ratio=0.5, alpha=0.3, beta=0.7, gamma=1.0):
        super(CombinedLoss, self).__init__()
        self.ce_ratio = ce_ratio
        self.ce = nn.CrossEntropyLoss()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, gamma=gamma)

    def forward(self, pred, target):
        loss_ce = self.ce(pred, target)
        loss_tversky = self.tversky(pred, target)
        return self.ce_ratio * loss_ce + (1 - self.ce_ratio) * loss_tversky
