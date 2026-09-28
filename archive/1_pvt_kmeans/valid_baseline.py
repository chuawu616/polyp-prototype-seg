# 檔案: valid_baseline.py
# 描述: 獨立的 Baseline 模型驗證腳本，包含內嵌 Dataset。

import os
import glob
import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import cv2

# --- 1. 導入模型定義 ---
# 假設 BaselineSegmenter.py 在 models 資料夾下
from models.BaselineSegmenter import BaselineSegmenter
from util.metric import SegmentationMetric # 復用 metric 類

random.seed(42)
# --- 2. 內嵌的 Dataset 定義 (獨立版) ---
class BaselineValDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(352, 352)):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        self.image_paths.extend(sorted(glob.glob(os.path.join(image_dir, '*.jpg'))))
        
        self.mask_map = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(mask_dir, '*.png'))
        }
        
        self.image_paths = [p for p in self.image_paths if os.path.splitext(os.path.basename(p))[0] in self.mask_map]
        print(f"找到 {len(self.image_paths)} 組驗證樣本。")
        
        self.transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = self.mask_map[base_name]

        img_pil = Image.open(img_path).convert("RGB")
        mask_pil = Image.open(mask_path).convert("L")
        
        # 圖像轉 Tensor
        image_tensor = self.transform(img_pil)
        
        # 遮罩轉 Numpy (保持原始尺寸)
        mask_np = (np.array(mask_pil) > 128).astype(np.uint8)
        
        # 原始圖像轉 Numpy (保持原始尺寸)
        original_image_np = np.array(img_pil)
        
        return image_tensor, mask_np, original_image_np, img_path

# --- 3. 自定義 Collate Function ---
def validation_collate(batch):
    # 批次化 Image Tensor
    images = torch.stack([item[0] for item in batch], 0)
    # 其他不規則數據收集為 list
    masks = [item[1] for item in batch]
    original_images = [item[2] for item in batch]
    paths = [item[3] for item in batch]
    return images, masks, original_images, paths

# --- 4. 視覺化函數 ---
def save_visualization(original_image, true_mask, pred_mask, output_path):
    FG_COLOR = [255, 255, 0] # 黃色
    
    if original_image.dtype != np.uint8:
        original_image = (original_image * 255).clip(0, 255).astype(np.uint8)
        
    h, w = original_image.shape[:2]
    
    # Resize masks to original size
    if pred_mask.shape != (h, w):
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    if true_mask.shape != (h, w):
        true_mask = cv2.resize(true_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
    true_vis = np.zeros_like(original_image)
    pred_vis = np.zeros_like(original_image)
    
    true_vis[true_mask == 1] = FG_COLOR
    pred_vis[pred_mask == 1] = FG_COLOR
    
    # Add titles
    def add_title(img, txt):
        img = cv2.copyMakeBorder(img, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.putText(img, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        return img

    vis = np.concatenate([
        add_title(cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR), "Original"),
        add_title(cv2.cvtColor(true_vis, cv2.COLOR_RGB2BGR), "Ground Truth"),
        add_title(cv2.cvtColor(pred_vis, cv2.COLOR_RGB2BGR), "Prediction")
    ], axis=1)
    
    cv2.imwrite(output_path, vis)

# --- 5. 主程序 ---
if __name__ == '__main__':
    # --- 設定參數 ---
    CONFIG = {
        'device': 'cuda:0',
        'model_path': './final_model_baseline/best_model_baseline.pth',
        'image_dir': '/home/U116med/data/polyp/TestDataset/test/images/',
        'mask_dir': '/home/U116med/data/polyp/TestDataset/test/masks/',
        'image_size': 352,
        'batch_size': 4,
        'output_dir': './val_results_baseline/',
        'num_visualize': 10
    }
    
    device = torch.device(CONFIG['device'])
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # --- 加載模型 ---
    print("正在加載 Baseline 模型...")
    model = BaselineSegmenter(pretrained=False).to(device)
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=device))
    model.eval()
    
    # --- 加載數據 ---
    val_dataset = BaselineValDataset(CONFIG['image_dir'], CONFIG['mask_dir'], CONFIG['image_size'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, 
                            num_workers=4, collate_fn=validation_collate)
    
    # --- 評估 ---
    metric = SegmentationMetric(num_classes=2)
    indices_to_save = random.sample(range(len(val_dataset)), min(len(val_dataset), CONFIG['num_visualize']))
    
    print("開始驗證...")
    with torch.no_grad():
        for i, (images, masks, original_images, paths) in enumerate(tqdm(val_loader)):
            images = images.to(device)
            
            # 前向傳播
            logits = model(images) # (B, 1, H, W)
            preds = (logits > 0).long().squeeze(1).cpu().numpy() # (B, H, W)
            
            # 批次處理
            for j in range(len(images)):
                true_mask = masks[j]
                pred_mask = preds[j]
                
                # Resize pred to GT size for metric calculation
                if pred_mask.shape != true_mask.shape:
                    pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8), 
                                                 (true_mask.shape[1], true_mask.shape[0]), 
                                                 interpolation=cv2.INTER_NEAREST)
                else:
                    pred_mask_resized = pred_mask

                metric.update(np.array([true_mask]), np.array([pred_mask_resized]))
                
                # 視覺化
                idx = i * CONFIG['batch_size'] + j
                if idx in indices_to_save:
                    save_path = os.path.join(CONFIG['output_dir'], f"{os.path.basename(paths[j])}")
                    save_visualization(original_images[j], true_mask, pred_mask, save_path)

    scores = metric.get_scores()
    
    # 獲取前景 Dice
    fg_dice = scores['Class_Dice'][1]
    
    # 獲取 Mean IoU (背景 IoU 和 前景 IoU 的平均)
    mean_iou = scores['Mean_IoU']
    
    # 獲取前景 IoU (通常比 mIoU 更能反映息肉分割的好壞)
    fg_iou = scores['Class_IoU'][1]

    print("\n" + "="*40)
    print(f"  Validation Dice (FG): {fg_dice:.4f}")
    print(f"  Validation mIoU:      {mean_iou:.4f}")
    print(f"  Validation IoU (FG):  {fg_iou:.4f}")
    print("="*40)
    print(f"視覺化結果已儲存至: {CONFIG['output_dir']}")