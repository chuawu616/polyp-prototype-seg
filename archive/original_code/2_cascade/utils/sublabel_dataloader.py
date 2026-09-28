import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch

class SublabelPolypDataset(data.Dataset):
    def __init__(self, image_root, gt_root, sublabel_root, trainsize, augmentations):
        self.trainsize = trainsize
        self.augmentations = augmentations
        
        # --- 1. Robust Filename Matching---
        try:
            images_list = os.listdir(image_root)
            gts_list = os.listdir(gt_root)
            subs_list = os.listdir(sublabel_root)
        except FileNotFoundError as e:
            print(f"Error loading datasets: {e}")
            exit()

        img_map = {os.path.splitext(f)[0]: f for f in images_list if f.lower().endswith(('.jpg', '.png', '.jpeg'))}
        gt_map = {os.path.splitext(f)[0]: f for f in gts_list if f.lower().endswith('.png')}
        sub_map = {os.path.splitext(f)[0]: f for f in subs_list if f.lower().endswith('.png')}
        
        common_names = sorted(list(set(img_map.keys()) & set(gt_map.keys()) & set(sub_map.keys())))
        
        print(f"[{self.__class__.__name__}] Alignment Report:")
        print(f"  - Images found: {len(img_map)}")
        print(f"  - GTs found:    {len(gt_map)}")
        print(f"  - Subs found:   {len(sub_map)}")
        print(f"  - Intersection: {len(common_names)} (Final Dataset Size)")
        
        if len(common_names) == 0:
            print("Error: No matched files found! Check your directory paths or filenames.")
            exit()

        # 重建嚴格對齊的路徑列表
        self.images = [os.path.join(image_root, img_map[n]) for n in common_names]
        self.gts = [os.path.join(gt_root, gt_map[n]) for n in common_names]
        self.sublabels = [os.path.join(sublabel_root, sub_map[n]) for n in common_names]
        
        self.size = len(self.images)

        # Image only transforms (Color normalization)
        self.img_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.gt_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        sublabel = self.binary_loader(self.sublabels[index]) 
        
        # --- Synchronized Augmentation ---
        seed = np.random.randint(2147483647)
        
        if self.augmentations == 'True':
            random.seed(seed); torch.manual_seed(seed)
            
            angle = transforms.RandomRotation.get_params([-90, 90])
            image = image.rotate(angle)
            gt = gt.rotate(angle)
            sublabel = sublabel.rotate(angle, resample=Image.NEAREST)
            
            if random.random() > 0.5:
                image = transforms.functional.vflip(image)
                gt = transforms.functional.vflip(gt)
                sublabel = transforms.functional.vflip(sublabel)
                
            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
                gt = transforms.functional.hflip(gt)
                sublabel = transforms.functional.hflip(sublabel)
                
            color_transform = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.015)
            image = color_transform(image)
            
        image = image.resize((self.trainsize, self.trainsize), Image.BILINEAR)
        gt = gt.resize((self.trainsize, self.trainsize), Image.NEAREST)
        sublabel = sublabel.resize((self.trainsize, self.trainsize), Image.NEAREST)
        
        image = self.img_normalize(image)
        gt = self.gt_tensor(gt)
        
        sublabel = torch.from_numpy(np.array(sublabel)).long()
        
        return image, gt, sublabel

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

    def __len__(self):
        return self.size

def get_sublabel_loader(image_root, gt_root, sublabel_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True, augmentation='False'):
    dataset = SublabelPolypDataset(image_root, gt_root, sublabel_root, trainsize, augmentation)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader