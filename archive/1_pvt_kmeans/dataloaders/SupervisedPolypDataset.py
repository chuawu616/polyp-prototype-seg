import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

class SupervisedPolypDataset(Dataset):
    def __init__(self, image_dir, mask_dir, prototype_dir=None, target_size=(352, 352), return_original=False):
        """
        初始化有監督的息肉資料集。

        Args:
            image_dir (str): 包含訓練影像的資料夾路徑。
            mask_dir (str): 包含對應的真實二元遮罩的資料夾路徑。
            target_size (tuple): 所有影像和遮罩將被統一調整到的目標尺寸。
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.return_original = return_original
        # 掃描影像檔案
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        self.image_paths.extend(sorted(glob.glob(os.path.join(image_dir, '*.jpg'))))
        self.prototype_dir = prototype_dir
        
        # 建立檔名到遮罩路徑的映射
        self.mask_map = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(mask_dir, '*.png'))
        }
        
        # 過濾掉沒有對應遮罩的影像
        self.image_paths = [p for p in self.image_paths if os.path.splitext(os.path.basename(p))[0] in self.mask_map]

        if not self.image_paths:
            raise FileNotFoundError(f"在影像目錄 {image_dir} 和遮罩目錄 {mask_dir} 之間找不到匹配的檔案。")
        print(f"找到 {len(self.image_paths)} 組有效的 影像-遮罩 對。")

        if self.prototype_dir:
            self.proto_map = {
                os.path.splitext(os.path.basename(p))[0].replace('_protos', ''): p
                for p in glob.glob(os.path.join(prototype_dir, '*.npy'))
            }        
            self.image_paths = [p for p in self.image_paths 
                                if os.path.splitext(os.path.basename(p))[0] in self.mask_map 
                                and os.path.splitext(os.path.basename(p))[0] in self.proto_map]

        if not self.image_paths:
            raise FileNotFoundError("找不到任何匹配的 影像-遮罩-原型 组合。")
        print(f"找到 {len(self.image_paths)} 組有效的 影像-遮罩-原型 組合。")

        # 定義影像的轉換流程
        self.image_transform = T.Compose([
            T.Resize(target_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 定義遮罩的轉換流程
        self.mask_transform = T.Compose([
            T.Resize(target_size, interpolation=T.InterpolationMode.NEAREST),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img_pil = Image.open(image_path).convert("RGB")
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = self.mask_map[base_name]
        mask_pil = Image.open(mask_path).convert("L")

        # 應用轉換
        image_tensor = self.image_transform(img_pil)
        
        mask_resized_pil = self.mask_transform(mask_pil)
        mask_np = np.array(mask_resized_pil)
        mask_binary = (mask_np > 128).astype(np.uint8)
        mask_tensor = torch.from_numpy(mask_binary).long()
        
        if self.prototype_dir:
            proto_path = self.proto_map[base_name]
            prototypes = np.load(proto_path)
            prototype_tensor = torch.from_numpy(prototypes).float()
        else:
            prototype_tensor = torch.empty(0) 
        
        if self.return_original:
            original_image_np = np.array(img_pil)
            return image_tensor, mask_tensor, prototype_tensor, original_image_np, image_path
        else:
            return image_tensor, mask_tensor, prototype_tensor

        