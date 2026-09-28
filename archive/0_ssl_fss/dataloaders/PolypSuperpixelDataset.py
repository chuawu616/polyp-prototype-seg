import os
import glob
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from dataloaders.dataset_utils import DATASET_INFO, get_normalize_op, read_image_rgb, read_label_gray
import cv2

class PolypSuperpixelDataset(Dataset):
    def __init__(self, image_dir, pseudolabel_dir, mode, transforms, num_rep=2, min_area_threshold=100, output_size=(256, 256)):
        super().__init__()
        
        assert num_rep % 2 == 0, "num_rep must be an even number for pairing."
        
        self.image_dir = image_dir
        self.pseudolabel_dir = pseudolabel_dir
        self.transforms = transforms
        self.num_rep = num_rep
        self.min_area_threshold = int(min_area_threshold)
        self.output_size = output_size

        dataset_name = 'POLYP'
        self.img_modality = DATASET_INFO[dataset_name]['MODALITY']
        self.pseu_label_name = DATASET_INFO[dataset_name]['PSEU_LABEL_NAME']
        self.nclass = len(self.pseu_label_name)

        self.img_lb_fids = self._organize_file_paths()
        if not self.img_lb_fids:
            raise FileNotFoundError(f"No matching image-pseudolabel pairs found in"
                                    f"\n  Image dir: {self.image_dir}"
                                    f"\n  Pseudolabel dir: {self.pseudolabel_dir}")
                                    
        self.norm_func = get_normalize_op(self.img_modality)
        self.dataset_samples = self._read_dataset()

    def _organize_file_paths(self):
        """掃描資料夾並配對影像和偽標籤的路徑。"""
        image_paths = sorted(glob.glob(os.path.join(self.image_dir, '*.png')))
        image_paths.extend(sorted(glob.glob(os.path.join(self.image_dir, '*.jpg')))) # 支援jpg
        
        out_list = []
        for img_path in image_paths:
            basename = os.path.basename(img_path)
            # 假設偽標籤命名為 'image_001_label.png' 或 'image_001.png' -> 'image_001_label.png'
            label_name = os.path.splitext(basename)[0] + '_label.png'
            lb_path = os.path.join(self.pseudolabel_dir, label_name)
            
            if os.path.exists(lb_path):
                out_list.append({"img_fid": img_path, "lbs_fid": lb_path})
            else:
                # 為了調試，印出找不到的檔案
                # print(f"Warning: Pseudolabel not found for {basename} at expected path {lb_path}")
                pass
        
        print(f"Found {len(out_list)} image-pseudolabel pairs.")
        return out_list

    def _read_dataset(self):
        """將所有2D影像和偽標籤讀入記憶體。"""
        out_list = []
        for path_dict in self.img_lb_fids:
            img = read_image_rgb(path_dict['img_fid'])
            lb = read_label_gray(path_dict['lbs_fid'])

            if img is None or lb is None:
                continue

            # --- 新增的 Resize 步骤 ---
            # 1. Resize 图像
            #    使用 cv2.INTER_LINEAR (双线性插值)
            #    注意：cv2.resize 的 target_size 是 (width, height)
            target_size_wh = (self.output_size[1], self.output_size[0])
            img_resized = cv2.resize(img, target_size_wh, interpolation=cv2.INTER_LINEAR)
            
            # 2. Resize 标签
            #    确保标签也是正确的尺寸，即使我们假设它已经是了
            #    使用 cv2.INTER_NEAREST (最近邻插值)
            if lb.shape != self.output_size:
                lb_resized = cv2.resize(lb, target_size_wh, interpolation=cv2.INTER_NEAREST)
            else:
                lb_resized = lb

            # 将标签扩展为 (H, W, 1)
            lb_resized = np.expand_dims(lb_resized, axis=-1)

            # 标准化 Resize 后的图像
            img_normalized = self.norm_func(img_resized)

            sample = {
                "image": img_normalized,  # (256, 256, 3), float32
                "label": lb_resized,      # (256, 256, 1), int32
                "sup_max_cls": lb_resized.max(),
                "filename": os.path.basename(path_dict['img_fid'])
            }
            out_list.append(sample)
        return out_list

    def _supcls_pick_binarize(self, super_map, sup_max_cls):
        """
        智能地選取一個superpixel作為前景。
        """
        # 如果 sup_max_cls < 1，說明這張圖本身就是空的，直接返回
        min_area_threshold = self.min_area_threshold
        if sup_max_cls < 1:
            return np.zeros_like(super_map, dtype=np.float32)

        unique_ids, counts = np.unique(super_map, return_counts=True)
        
        valid_indices = (unique_ids != 0) & (counts >= min_area_threshold)
        valid_ids = unique_ids[valid_indices]
        valid_areas = counts[valid_indices]

        if len(valid_ids) == 0:
            # 如果過濾後沒有合格的區域，就退回到“愚蠢”採樣模式，但只從ID大於0的裡面選
            # 這樣至少能保證選出一個非空的區域，即使它很小
            valid_ids = unique_ids[unique_ids != 0]
            if len(valid_ids) == 0: # 極端情況下，連非0的ID都沒有
                 return np.zeros_like(super_map, dtype=np.float32)
            random_id = random.choice(valid_ids)
        else:
            # 根據面積進行加權隨機採樣
            weights = valid_areas
            random_id = random.choices(population=valid_ids, weights=weights, k=1)[0]
        
        binary_mask = (super_map == random_id).astype(np.float32)
        return binary_mask

    def getMaskMedImg(self, label, class_id, class_ids):
        """
        Generate FG/BG mask from the segmentation mask

        Args:
            label:          semantic mask
            class_id:       semantic class of interest
            class_ids:      all class id in this episode #基本上好像也只有0, 1，不影響先不動
        """
        #torch.where(condition, x, y)：條件為 True → 取 x，False → 取 y
        fg_mask = torch.where(label == class_id, torch.ones_like(label), torch.zeros_like(label))
        bg_mask = torch.where(label != class_id, torch.ones_like(label), torch.zeros_like(label))
        #把類別0的也排除掉
        for cid in class_ids:
            if cid != class_id:
                bg_mask[label == cid] = 0
        return {'fg_mask': fg_mask, 'bg_mask': bg_mask}

    def __len__(self):
        return len(self.dataset_samples)

    def __getitem__(self, index):
        index = index % len(self.dataset_samples)
        sample_dict = self.dataset_samples[index]
        
        sup_max_cls = sample_dict['sup_max_cls']
        if sup_max_cls < 1:
            return self.__getitem__(random.randint(0, len(self) - 1))

        # --- 重要：这里的 image 和 label 已经是 256x256 了 ---
        # _supcls_pick_binarize 返回的 label_t 也是 256x256
        label_t = self._supcls_pick_binarize(sample_dict["label"], sup_max_cls)
        
        # --- 现在 concatenate 可以成功执行 ---
        composite_img = np.concatenate([sample_dict["image"], label_t], axis=-1)
        
        augmented_pairs = []
        for _ in range(self.num_rep):
            aug_img, aug_lb = self.transforms(composite_img, c_img=3, c_label=1, nclass=8, use_onehot=False) # 假设最多8类
            aug_img_t = torch.from_numpy(np.transpose(aug_img, (2, 0, 1))).float()
            aug_lb_t = torch.from_numpy(aug_lb.squeeze(-1)).long()
            augmented_pairs.append({"image": aug_img_t, "label": aug_lb_t})

        support_images, support_masks, query_images, query_labels = [], [], [], []
        pseudo_class_id = 1
        class_ids = [pseudo_class_id]

        for i, item in enumerate(augmented_pairs):
            if i % 2 == 0:
                support_images.append(item["image"])
                support_masks.append(self.getMaskMedImg(item["label"], pseudo_class_id, class_ids))
            else:
                query_images.append(item["image"])
                query_labels.append(item["label"])

        return {
            'class_ids': class_ids,
            'support_images': [support_images],
            'support_mask': [support_masks],
            'query_images': query_images,
            'query_labels': query_labels,
            'support_inst': [],
            'support_scribbles': [],
            'query_masks': [],
            'query_cls_idx': []
        }