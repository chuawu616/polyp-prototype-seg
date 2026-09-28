# 檔案: dataloaders/SupervisedContrastiveDataset.py

import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
class SupervisedContrastiveDataset(Dataset):
    """
    用於第二階段監督對比學習的資料集。
    """
    def __init__(self, image_dir, sublabel_dir, image_size=(352, 352)):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        self.image_paths.extend(sorted(glob.glob(os.path.join(image_dir, '*.jpg'))))
        
        self.sublabel_map = {os.path.splitext(os.path.basename(p))[0]: p
                           for p in glob.glob(os.path.join(sublabel_dir, '*.png'))}
        
        self.image_paths = [p for p in self.image_paths if os.path.splitext(os.path.basename(p))[0] in self.sublabel_map]
        
        # 增强流程 (这次可以包含几何变换，因为我们有标签来对齐)
        self.image_transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.label_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15, interpolation=T.InterpolationMode.NEAREST),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        sublabel_path = self.sublabel_map[base_name]

        img_pil = Image.open(img_path).convert("RGB")
        sublabel_pil = Image.open(sublabel_path)
        
        # 為了讓圖像和標籤應用相同的隨機幾何變換，需要同步種子
        seed = torch.seed()
        
        torch.manual_seed(seed)
        view1 = self.image_transform(img_pil)
        torch.manual_seed(seed)
        sublabel1 = self.label_transform(sublabel_pil)
        sublabel1 = torch.from_numpy(np.array(sublabel1)).long()

        torch.manual_seed(seed + 1) # 使用不同的種子生成第二個視圖
        view2 = self.image_transform(img_pil)
        torch.manual_seed(seed + 1)
        sublabel2 = self.label_transform(sublabel_pil)
        sublabel2 = torch.from_numpy(np.array(sublabel2)).long()

        return view1, sublabel1, view2, sublabel2