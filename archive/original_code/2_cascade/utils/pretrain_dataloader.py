import os
from PIL import Image, ImageFilter
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch

class GaussianBlur(object):
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class PretrainPolypDataset(data.Dataset):
    def __init__(self, image_root, trainsize=352, augmentation='True', mode='contrastive'):
        self.trainsize = trainsize
        self.augmentations = augmentation
        self.mode = mode
        
        self.images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) 
                              if f.endswith('.jpg') or f.endswith('.png')])
        self.size = len(self.images)
        print(f"Dataset loaded: mode={mode}, size={self.size}")

        # 幾何變換 (需對齊)
        self.geometric_transform = transforms.Compose([
            transforms.RandomRotation(90, expand=False),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((self.trainsize, self.trainsize)),
        ])

        # 光度變換 (獨立隨機)
        self.photometric_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.simple_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        
        if self.mode == 'contrastive':
            seed_geo = np.random.randint(2147483647)
            
            # 確保兩個 View 幾何一致
            random.seed(seed_geo); torch.manual_seed(seed_geo)
            img_geo = self.geometric_transform(image)
            
            # 光度獨立
            view1 = self.photometric_transform(img_geo)
            view2 = self.photometric_transform(img_geo)
            
            return view1, view2
        else:
            # Inference mode (只返回單張圖)
            image = self.simple_transform(image)
            return image

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __len__(self):
        return self.size