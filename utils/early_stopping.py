import numpy as np
import torch


class EarlyStopping:
    """早停机制，当验证指标不再改善时停止训练"""

    def __init__(self, patience=10, min_delta=1e-4, mode='max'):
        """
        Args:
            patience: 容忍多少个epoch没有改善
            min_delta: 最小改善幅度
            mode: 'max' 表示指标越大越好，'min' 表示越小越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        """
        Returns:
            True: 应该停止训练
            False: 继续训练
        """
        if self.best_score is None:
            self.best_score = val_score
            return False

        if self.mode == 'max':
            if val_score > self.best_score + self.min_delta:
                self.best_score = val_score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'min'
            if val_score < self.best_score - self.min_delta:
                self.best_score = val_score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True

        return False

    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
