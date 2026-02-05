from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import random

# ================== 修改 1: 适配肺实质二分类映射 ==================
# 0: 背景 -> 类别 0
# 255: 肺实质 -> 类别 1
label_mapping = {
    0: 0,
    255: 1
}
# 逆向映射：用于可视化将类别 1 变回 255 像素值
inverse_mapping = {0: 0, 1: 255}


class CustomImageDataset(Dataset):
    def __init__(self, data_type, transform=None):
        # ================== 修改 2: 更新数据根目录 ==================
        self.img_dir = r'D:\PyCharm\pytorch\UNet-LiTS2017-main\RP\RP'
        self.data_type = data_type
        self.transform = transform
        self.image_paths = []
        self.label_paths = []

        # 注意：确保你的 txt 文件放在正确的位置（这里默认在 ./ 目录下）
        txt_path = './preprocessRP/'+ self.data_type + '.txt'

        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"找不到索引文件: {txt_path}, 请检查第一步划分代码是否成功运行。")

        with open(txt_path, 'r') as txt_file:
            for row in txt_file:
                case_name = row.strip()
                image_case_dir = os.path.join(self.img_dir, case_name, 'Image')
                label_case_dir = os.path.join(self.img_dir, case_name, 'GT')

                if os.path.exists(image_case_dir):
                    for case in os.listdir(image_case_dir):
                        self.image_paths.append(os.path.join(image_case_dir, case))
                        self.label_paths.append(os.path.join(label_case_dir, case))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # 读取图像和标签
        image = Image.open(img_path).convert("RGB")  # 即使是灰度图，转为RGB可适配预训练模型
        label = Image.open(label_path).convert("L")

        if self.transform:
            # 同步变换：确保图像和标签的旋转、翻转完全一致
            seed = 3407 + idx
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.transform(image)

            random.seed(seed)
            torch.manual_seed(seed)
            label = self.transform(label)

        # 图像转 Tensor: (C, H, W), 归一化到 [0, 1]
        image = transforms.ToTensor()(image)

        # 标签处理
        label_np = np.array(label)
        # ================== 修改 3: 二分类映射逻辑 ==================
        # 将像素值 255 映射为类别 1
        label_mapped = np.zeros_like(label_np, dtype=np.int64)
        label_mapped[label_np > 128] = 1  # 考虑到插值，使用阈值更稳健
        label = torch.from_numpy(label_mapped).long()  # (H, W)

        return image, label


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    set_seed(3407)

    # 定义数据增强
    # 注意：对于标签，插值必须使用 NEAREST，否则会产生不存在的中间类别像素
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        # transforms.RandomRotation(15), # 肺部一般不建议大角度旋转，可根据需求开启
    ])

    # 实例化
    try:
        train_dataset = CustomImageDataset(data_type='train', transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        print(f'成功加载训练集，切片数量: {len(train_dataset)}')

        # 可视化校验
        images, labels = next(iter(train_loader))

        # 准备可视化：将 [0, 1] 类别转回 [0, 255] 像素
        # labels 是 (B, H, W)，转为 (B, 1, H, W) 后再 stack 成 RGB 方便显示
        vis_labels = (labels.float() * 255).unsqueeze(1).byte()
        vis_labels = torch.cat([vis_labels] * 3, dim=1)

        grid_img = make_grid(images)
        grid_label = make_grid(vis_labels.float() / 255.0)  # make_grid 期望 float [0,1]

        plt.figure(figsize=(15, 8))
        plt.subplot(1, 2, 1);
        plt.title("CT Images (Windowed)")
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.subplot(1, 2, 2);
        plt.title("Lung Masks (GT)")
        plt.imshow(grid_label.permute(1, 2, 0))
        plt.show()

    except Exception as e:
        print(f"出错: {e}")
