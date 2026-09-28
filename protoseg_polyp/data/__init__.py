"""Polyp datasets (PraNet split). Transforms are identical to the original research code.

Layout:  <root>/images/*.png|jpg   <root>/masks/*.png   [<sp_root>/<stem>.npy]
"""
import os
import random

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _list(root, exts):
    return sorted(os.path.join(root, f) for f in os.listdir(root) if f.endswith(exts))


def _rgb(path):
    with open(path, 'rb') as f:
        return Image.open(f).convert('RGB')


def _gray(path):
    with open(path, 'rb') as f:
        return Image.open(f).convert('L')


def list_pairs(root):
    """Sorted (image, mask) paths of a PraNet-style folder, dropping pairs whose sizes differ."""
    images, gts = _list(os.path.join(root, 'images'), ('.jpg', '.png')), _list(os.path.join(root, 'masks'), '.png')
    assert len(images) == len(gts)
    keep = [(i, g) for i, g in zip(images, gts) if Image.open(i).size == Image.open(g).size]
    return [k[0] for k in keep], [k[1] for k in keep]


def split_train_val(root, val_fraction, seed=0):
    """Deterministic random split of the training folder into (train indices, val indices)."""
    n = len(list_pairs(root)[0])
    perm = np.random.RandomState(seed).permutation(n)
    n_val = int(round(n * val_fraction))
    return sorted(perm[n_val:].tolist()), sorted(perm[:n_val].tolist())


class PolypDataset(data.Dataset):
    """Returns (image, gt, sp_map). sp_map is a (H, W) long superpixel index map, or all -1 when unused.

    augmentation=True: random rotation (+-90), vertical / horizontal flip, colour jitter (image only).
    The same random seed is replayed for image, mask and superpixel map so they stay aligned.
    """

    def __init__(self, root, trainsize=352, augmentation=False, sp_root=None, indices=None):
        self.trainsize = trainsize
        images, gts = list_pairs(root)
        if indices is not None:
            images, gts = [images[i] for i in indices], [gts[i] for i in indices]
        self.images, self.gts = images, gts
        self.sp_paths = None if sp_root is None else [
            os.path.join(sp_root, os.path.splitext(os.path.basename(p))[0] + '.npy') for p in self.images]

        geo = [T.RandomRotation(90, expand=False, center=None, fill=None),
               T.RandomVerticalFlip(p=0.5), T.RandomHorizontalFlip(p=0.5)] if augmentation else []
        self.img_transform = T.Compose(
            geo + ([T.ColorJitter(0.1, 0.1, 0.1, 0.015)] if augmentation else []) +
            [T.Resize((trainsize, trainsize)), T.ToTensor(), T.Normalize(MEAN, STD)])
        self.gt_transform = T.Compose(geo + [T.Resize((trainsize, trainsize)), T.ToTensor()])
        sp_geo = [T.RandomRotation(90, expand=False, fill=0),
                  T.RandomVerticalFlip(p=0.5), T.RandomHorizontalFlip(p=0.5)] if augmentation else []
        self.sp_transform = T.Compose(sp_geo + [T.Resize((trainsize, trainsize),
                                                         interpolation=T.InterpolationMode.NEAREST)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        seed = np.random.randint(2147483647)
        random.seed(seed); torch.manual_seed(seed)
        image = self.img_transform(_rgb(self.images[index]))
        random.seed(seed); torch.manual_seed(seed)
        gt = self.gt_transform(_gray(self.gts[index]))

        if self.sp_paths is not None and os.path.exists(self.sp_paths[index]):
            sp = Image.fromarray(np.load(self.sp_paths[index]).astype(np.int32), mode='I')
            random.seed(seed); torch.manual_seed(seed)
            sp = torch.from_numpy(np.array(self.sp_transform(sp), dtype=np.int32)).long()
        else:
            sp = torch.full((self.trainsize, self.trainsize), -1, dtype=torch.long)
        return image, gt, sp


def get_loader(root, batchsize=16, trainsize=352, augmentation=False, sp_root=None, num_workers=4, shuffle=True,
               indices=None):
    ds = PolypDataset(root, trainsize, augmentation, sp_root, indices)
    return data.DataLoader(ds, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


class TestDataset:
    """Iterates (image tensor (1,3,S,S), gt PIL image at original size, file name)."""
    __test__ = False  # not a pytest test class

    def __init__(self, root, testsize=352, pairs=None):
        if pairs is None:
            self.images = _list(os.path.join(root, 'images'), ('.jpg', '.png'))
            self.gts = _list(os.path.join(root, 'masks'), ('.tif', '.png'))
        else:  # explicit (images, masks) lists, e.g. a validation split of the training folder
            self.images, self.gts = pairs
        self.transform = T.Compose([T.Resize((testsize, testsize)), T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        for img, gt in zip(self.images, self.gts):
            name = os.path.basename(img)
            yield self.transform(_rgb(img)).unsqueeze(0), _gray(gt), os.path.splitext(name)[0] + '.png'
