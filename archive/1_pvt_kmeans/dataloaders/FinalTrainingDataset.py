# 檔案: dataloaders/FinalTrainingDataset.py

import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A # 引入 Albumentations
from albumentations.pytorch import ToTensorV2

class FinalTrainingDataset(Dataset):
    """
    用於第三階段最終訓練的資料集。
    包含強大的 Albumentations 數據增強。
    """
    def __init__(self, image_dir, binary_mask_dir, sublabel_dir, image_size=(352, 352)):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*')))
        # 過濾非圖片文件
        self.image_paths = [p for p in self.image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]

        self.binary_mask_map = {os.path.splitext(os.path.basename(p))[0]: p
                                for p in glob.glob(os.path.join(binary_mask_dir, '*'))}
        
        self.sublabel_map = {os.path.splitext(os.path.basename(p))[0]: p
                             for p in glob.glob(os.path.join(sublabel_dir, '*'))}
        
        # 過濾
        self.image_paths = [p for p in self.image_paths 
                            if os.path.splitext(os.path.basename(p))[0] in self.binary_mask_map
                            and os.path.splitext(os.path.basename(p))[0] in self.sublabel_map]
        
        if not self.image_paths:
            raise FileNotFoundError("找不到任何匹配的 影像-二元遮罩-子標籤 組合。")
        print(f"找到 {len(self.image_paths)} 組有效的訓練樣本。")

        # --- 定義強大的 Augmentation Pipeline ---
        self.transform = A.Compose([
            # 1. 幾何變換 (Geometry) - 這些變換必須同時應用於 Image, Binary Mask, Sublabel
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5), # 隨機旋轉 +/- 30度
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=0, p=0.5), # 平移和縮放
            
            # 彈性變換與網格畸變 (對醫學影像特別有效，模擬器官形變)
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5),
            ], p=0.3),

            # 2. 像素級變換 (Pixel-level) - 只應用於 Image，不影響 Mask
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.MotionBlur(p=0.5),
                A.MedianBlur(blur_limit=3, p=0.5),
                A.Blur(blur_limit=3, p=0.5),
            ], p=0.3),
            
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            
            A.HueSaturationValue(p=0.3), # 顏色抖動
            
            # 3. 特殊增強 (CoarseDropout / Cutout)
            # 隨機挖掉一些小洞，強迫模型學習全局上下文，防止過擬合
            # 注意：這裡我們設置 mask_fill_value=0，讓 mask 在挖掉的地方變成背景
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, 
                            fill_value=0, mask_fill_value=0, p=0.3),

            # 4. 歸一化與轉 Tensor
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], 
        # 關鍵設置：定義額外的 mask 目標
        # 'mask' 是默認的目標類型，會自動應用幾何變換但不會應用像素級變換
        additional_targets={'mask0': 'mask', 'mask1': 'mask'} 
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # 讀取圖像 (轉為 NumPy 數組)
        img = np.array(Image.open(img_path).convert("RGB"))
        binary_mask = np.array(Image.open(self.binary_mask_map[base_name]).convert("L"))
        sublabel = np.array(Image.open(self.sublabel_map[base_name]).convert("L"))
        
        # 應用 Augmentation
        # 我們將 binary_mask 作為標準 'mask'，sublabel 作為 'mask1'
        augmented = self.transform(image=img, mask=binary_mask, mask1=sublabel)
        
        image_tensor = augmented['image']
        
        # 處理增強後的 Mask
        # Binary Mask 需要二值化 (Albumentations resize 默認使用 nearest，但為了保險)
        binary_mask_tensor = augmented['mask']
        
        # 二值化邏輯：如果是 Tensor，直接比較並轉換類型
        # 假設 mask 值範圍是 0-255 (float 或 uint8 tensor)
        if isinstance(binary_mask_tensor, torch.Tensor):
            # Tensor 操作
            binary_mask_tensor = (binary_mask_tensor > 128).long()
        else:
            # 如果為了某種原因它還是 numpy array (例如沒用 ToTensorV2)，則保留原邏輯
            binary_mask_tensor = torch.from_numpy((binary_mask_tensor > 128).astype(np.uint8)).long()
        
        # Sublabel
        sublabel_tensor = augmented['mask1'].long()
        
        return image_tensor, binary_mask_tensor, sublabel_tensor