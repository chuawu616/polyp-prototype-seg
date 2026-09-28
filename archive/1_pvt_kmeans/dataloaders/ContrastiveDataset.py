# 檔案: dataloaders/ContrastiveDataset.py

import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class ContrastiveDataset(Dataset):
    """
    用於像素級對比學習的資料集。
    每次返回同一張圖像的兩個不同增強版本 (視圖)。
    """
    def __init__(self, image_dir, image_size=(352, 352)):
        """
        Args:
            image_dir (str): 包含訓練影像的資料夾路徑。
            image_size (tuple): 圖像的目標尺寸。
        """
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        self.image_paths.extend(sorted(glob.glob(os.path.join(image_dir, '*.jpg'))))
        
        if not self.image_paths:
            raise FileNotFoundError(f"在 {image_dir} 中找不到任何圖像。")
        print(f"找到 {len(self.image_paths)} 張圖像用於對比學習预訓練。")
        
        # --- 核心：定義數據增強流程 ---
        # 按照您的要求，這裡只包含非幾何變換
        # 例如：顏色抖動、高斯模糊、灰度化、亮度/對比度調整
        self.transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            # 隨機應用顏色抖動
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            # 隨機應用灰度化
            T.RandomGrayscale(p=0.2),
            # 隨機應用高斯模糊
            T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            # 轉換為 Tensor 並進行標準化
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img_pil = Image.open(image_path).convert("RGB")
        
        # --- 核心：對同一張圖像應用兩次增強 ---
        view1 = self.transform(img_pil)
        view2 = self.transform(img_pil)
        
        return view1, view2