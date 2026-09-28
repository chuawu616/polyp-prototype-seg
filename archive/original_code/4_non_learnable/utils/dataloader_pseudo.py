import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch


class PolypDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize, augmentations,
                 sp_root=None, percentage=1.):
        self.trainsize    = trainsize
        self.augmentations = augmentations
        self.sp_root      = sp_root
        print(self.augmentations)

        self.images_all = sorted([
            image_root + f for f in os.listdir(image_root)
            if f.endswith('.jpg') or f.endswith('.png')
        ])
        self.gts_all = sorted([
            gt_root + f for f in os.listdir(gt_root)
            if f.endswith('.png')
        ])

        N   = len(self.images_all)
        K   = int(N * percentage)
        idx = np.linspace(0, N - 1, K, dtype=int)
        self.images = [self.images_all[i] for i in idx]
        self.gts    = [self.gts_all[i]    for i in idx]

        # Superpixel path list：stem 對應同名 .npy
        if self.sp_root is not None:
            self.sp_paths = [
                os.path.join(self.sp_root,
                             os.path.splitext(os.path.basename(p))[0] + '.npy')
                for p in self.images
            ]
        else:
            self.sp_paths = None

        self.filter_files()
        self.size = len(self.images)
        print(f'Training samples: {self.__len__()}  |  '
              f'Superpixel: {"ON" if self.sp_root else "OFF"}')

        # ── Image transforms ───────────────────────────────────────────────
        if self.augmentations == 'True':
            print('Augmentation: RandomRotation + RandomFlip + ColorJitter')

            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                       saturation=0.1, hue=0.015),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()
            ])
            # Superpixel：只有幾何變換，全部使用 NEAREST 避免 index 插值
            self.sp_transform = transforms.Compose([
                transforms.RandomRotation(90, expand=False, fill=0),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize(
                    (self.trainsize, self.trainsize),
                    interpolation=transforms.InterpolationMode.NEAREST
                )
            ])
        else:
            print('No augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()
            ])
            self.sp_transform = transforms.Compose([
                transforms.Resize(
                    (self.trainsize, self.trainsize),
                    interpolation=transforms.InterpolationMode.NEAREST
                )
            ])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt    = self.binary_loader(self.gts[index])

        # 同一個 seed 確保 image / gt / sp_map 的幾何增強完全一致
        seed = np.random.randint(2147483647)

        random.seed(seed); torch.manual_seed(seed)
        image = self.img_transform(image)   # (3, H, W) float

        random.seed(seed); torch.manual_seed(seed)
        gt = self.gt_transform(gt)          # (1, H, W) float [0,1]

        if self.sp_paths is not None and os.path.exists(self.sp_paths[index]):
            sp_np  = np.load(self.sp_paths[index]).astype(np.int32)  # (H, W)
            # PIL mode 'I'（32-bit signed int）用於幾何變換
            sp_pil = Image.fromarray(sp_np, mode='I')

            random.seed(seed); torch.manual_seed(seed)
            sp_pil = self.sp_transform(sp_pil)

            # PIL 'I' → numpy → long tensor
            sp_map = torch.from_numpy(
                np.array(sp_pil, dtype=np.int32)
            ).long()  # (H, W)
        else:
            # sp_root 未指定或檔案不存在：填 -1 當作無效 superpixel
            sp_map = torch.full(
                (self.trainsize, self.trainsize), -1, dtype=torch.long
            )

        return image, gt, sp_map

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images, gts, sp_paths = [], [], []
        sp_list = self.sp_paths if self.sp_paths else [None] * len(self.images)

        for img_path, gt_path, sp_path in zip(self.images, self.gts, sp_list):
            img = Image.open(img_path)
            gt  = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
                if sp_path is not None:
                    sp_paths.append(sp_path)

        self.images = images
        self.gts    = gts
        if self.sp_paths is not None:
            self.sp_paths = sp_paths

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return (img.resize((w, h), Image.BILINEAR),
                    gt.resize((w, h), Image.NEAREST))
        return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, trainsize,
               shuffle=True, num_workers=4, pin_memory=True,
               augmentation=False, percentage=1.,
               sp_root=None):
    """
    Args:
        sp_root: GT-guided superpixel .npy 目錄（None = 不使用）
    """
    dataset = PolypDataset(
        image_root, gt_root, trainsize, augmentation,
        sp_root=sp_root, percentage=percentage
    )
    return data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = sorted([
            image_root + f for f in os.listdir(image_root)
            if f.endswith('.jpg') or f.endswith('.png')
        ])
        self.gts = sorted([
            gt_root + f for f in os.listdir(gt_root)
            if f.endswith('.tif') or f.endswith('.png')
        ])
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size  = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt    = self.binary_loader(self.gts[self.index])
        name  = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')