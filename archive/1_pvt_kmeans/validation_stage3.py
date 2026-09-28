# 檔案: validate.py
# 描述: 一個獨立的、簡潔的腳本，用於評估 MetricSegmenter 模型的性能。

import os
import glob
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import cv2
from skimage import color

# --- 1. 導入模型定義 ---
# 確保這些檔案在您的 Python 路徑中
from models.MetricSegmenter import MetricSegmenter
from util.metric import SegmentationMetric

# --- 2. 內嵌的 Dataset 定義 ---
class ValidationDataset(Dataset):
    """
    一個專為此驗證腳本設計的簡化版 Dataset。
    """
    def __init__(self, image_dir, mask_dir, image_size=(352, 352)):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        self.image_paths.extend(sorted(glob.glob(os.path.join(image_dir, '*.jpg'))))
        
        self.mask_map = {os.path.splitext(os.path.basename(p))[0]: p
                       for p in glob.glob(os.path.join(mask_dir, '*.png'))}
        
        self.image_paths = [p for p in self.image_paths if os.path.splitext(os.path.basename(p))[0] in self.mask_map]
        
        self.image_transform = T.Compose([
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
        
        image_tensor = self.image_transform(img_pil)
        
        # 遮罩直接返回 NumPy 陣列
        mask_np = (np.array(mask_pil) > 128).astype(np.uint8)
        
        # 同時返回未經 resize 的原始圖像，用於視覺化
        original_image_np = np.array(img_pil)
        
        return image_tensor, mask_np, original_image_np, img_path

# --- 3. 視覺化函數 ---
def save_visualization(original_image, true_mask, binary_pred_mask, output_path):
    FG_COLOR_RGB = [255, 255, 0] # 黄色
    h, w, _ = original_image.shape

    # 將所有遮罩 resize 到原始圖像尺寸
    if binary_pred_mask.shape != (h, w):
        binary_pred_mask = cv2.resize(binary_pred_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    if true_mask.shape != (h, w):
        true_mask = cv2.resize(true_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
    true_colored = np.zeros_like(original_image)
    binary_pred_colored = np.zeros_like(original_image)
    true_colored[true_mask == 1] = FG_COLOR_RGB
    binary_pred_colored[binary_pred_mask == 1] = FG_COLOR_RGB

    font = cv2.FONT_HERSHEY_SIMPLEX
    def add_title(image, text):
        img_with_title = cv2.copyMakeBorder(image, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.putText(img_with_title, text, (10, 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        return img_with_title

    img1 = add_title(cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR), "Original")
    img2 = add_title(cv2.cvtColor(true_colored, cv2.COLOR_RGB2BGR), "Ground Truth")
    img3 = add_title(cv2.cvtColor(binary_pred_colored, cv2.COLOR_RGB2BGR), "Prediction")

    combined_image = np.concatenate((img1, img2, img3), axis=1)
    cv2.imwrite(output_path, combined_image)

# --- 4. 主執行流程 ---
if __name__ == '__main__':
    
    # --- 在此處配置您的驗證 ---
    CONFIG = {
        'device': 'cuda:0',
        'reload_model_path': './final_model_metric/best_model_stage3_mlw005.pth',
        'val_image_dir': '/home/U116med/data/polyp/TestDataset/test/images/',
        'val_mask_dir': '/home/U116med/data/polyp/TestDataset/test/masks/',
        'image_size': 352,
        'val_batch_size': 4,
        
        # 模型配置 (必須與訓練時完全一致)
        'model_cfg': {
            'decoder_out_channels': 256,
            'num_fg_prototypes': 4,
        },
        
        # 輸出配置
        'output_dir': './validation_results/',
        'num_visualizations': 10,
    }
    # --------------------------

    # --- 準備工作 ---
    device = torch.device(CONFIG['device'])
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # --- 載入模型 ---
    print(f"正在從 {CONFIG['reload_model_path']} 載入模型...")
    model = MetricSegmenter(cfg=CONFIG['model_cfg']).to(device)
    try:
        model.load_state_dict(torch.load(CONFIG['reload_model_path'], map_location=device))
    except Exception as e:
        print(f"加載模型失敗: {e}")
        exit()
    model.eval()

    # --- 載入資料 ---
    print("正在載入驗證資料集...")
    val_dataset = ValidationDataset(
        image_dir=CONFIG['val_image_dir'],
        mask_dir=CONFIG['val_mask_dir'],
        image_size=CONFIG['image_size']
    )
    # 自定義 collate_fn 以處理尺寸不一的原始圖像
    def validation_collate_fn(batch):
        images = torch.stack([item[0] for item in batch], 0)
        masks = [item[1] for item in batch] # 返回 list of np.array
        original_images = [item[2] for item in batch]
        paths = [item[3] for item in batch]
        return images, masks, original_images, paths

    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['val_batch_size'], 
        shuffle=False, 
        num_workers=4,
        collate_fn=validation_collate_fn
    )

    # --- 評估 ---
    print("###### 開始驗證 ######")
    metric = SegmentationMetric(num_classes=2)
    indices_to_save = np.random.choice(len(val_dataset), 
                                       min(CONFIG['num_visualizations'], len(val_dataset)), 
                                       replace=False)

    with torch.no_grad():
        for i, (images, masks_list, original_images_list, paths) in enumerate(tqdm(val_loader, desc="驗證中")):
            images = images.to(device)
            
            # 模型預測
            binary_logits, _ = model(images)
            pred_masks_tensor = binary_logits.argmax(dim=1).cpu()
            
            # 遍歷批次中的每個樣本
            for j in range(images.shape[0]):
                pred_mask = pred_masks_tensor[j].numpy()
                true_mask = masks_list[j]
                
                # 更新指標 (需要將真實遮罩也 resize 到模型輸出尺寸)
                h, w = pred_mask.shape
                if true_mask.shape != (h, w):
                    true_mask_resized = cv2.resize(true_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                else:
                    true_mask_resized = true_mask

                metric.update(np.expand_dims(true_mask_resized, 0), np.expand_dims(pred_mask, 0))

                # 儲存視覺化結果
                current_index = i * CONFIG['val_batch_size'] + j
                if current_index in indices_to_save:
                    base_name = os.path.splitext(os.path.basename(paths[j]))[0]
                    output_path = os.path.join(CONFIG['output_dir'], f"result_{base_name}.png")
                    save_visualization(
                        original_image=original_images_list[j],
                        true_mask=true_mask,
                        binary_pred_mask=pred_mask,
                        output_path=output_path
                    )
    
    # --- 打印結果 ---
    scores = metric.get_scores()
    foreground_dice = scores["Class_Dice"][1]
    mean_iou = scores["Mean_IoU"]
    
    print("\n" + "="*50)
    print("             驗證完成")
    print("="*50)
    print(f"前景 (息肉) 的 Dice 分數: {foreground_dice:.4f}")
    print(f"平均 IoU (mIoU):          {mean_iou:.4f}")
    print("="*50)
    print(f"視覺化結果已儲存至: {CONFIG['output_dir']}")