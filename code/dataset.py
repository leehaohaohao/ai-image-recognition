import os
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np

class FFDIDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None, max_samples=None):
        self.img_path = []
        self.img_label = []
        self.transform = transform if transform is not None else None

        count = 0  # 总计数
        for path, label in zip(img_path, img_label):
            self.img_path.append(path)
            self.img_label.append(label)
            count += 1

            # 达到限制数量则停止
            if max_samples is not None and count >= max_samples:
                break

    def __getitem__(self, index, device=None):
        path = self.img_path[index]
        label = self.img_label[index]

        if not os.path.exists(path):  # 检查图片路径是否存在
            print(f"Warning: File not found - {path}, skipping.")
            # 返回一个空图像或默认值，防止中断
            img = Image.new('RGB', (256, 256))  # 创建一个空白图像
        else:
            try:
                img = Image.open(path).convert('RGB')  # 加载图片
            except Exception as e:
                print(f"Error loading image: {path}, skipping. Error: {e}")
                img = Image.new('RGB', (256, 256))  # 创建一个空白图像

        if self.transform is not None:
            img = self.transform(img)

        if device:
            img = img.to(device)  # 将数据迁移到设备

        return img, torch.tensor(label, dtype=torch.float32).to(device)  # 将标签迁移到设备

    def __len__(self):
        return len(self.img_path)
