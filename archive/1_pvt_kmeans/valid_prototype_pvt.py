import os
import glob
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
import cv2
import matplotlib.pyplot as plt
import random
from skimage import color
import matplotlib.cm as cm

from models.PrototypeSegmenter import PrototypeSegmenter, FPNDecoder
from dataloaders.SupervisedPolypDataset import SupervisedPolypDataset
from util.metric import SegmentationMetric
from config import ex # 導入統一的設定檔

# --- 視覺化輔助函式 ---
def save_segmentation_results(original_image, true_mask, binary_pred_mask, multiclass_pred_mask, output_path):
    """
    將原圖、真實遮罩、二元預測、多類別預測拼接並儲存 (使用高對比度顏色)。
    """
    # ... (确保 original_image 是 uint8 的逻辑不变) ...
    if original_image.dtype != np.uint8:
        original_image = (original_image * 255).clip(0, 255).astype(np.uint8)
    if len(original_image.shape) == 3 and original_image.shape[0] < 5:
        original_image = np.transpose(original_image, (1, 2, 0))

    h, w, _ = original_image.shape

    # --- 尺寸统一的逻辑保持不变 ---
    # ... (确保所有 mask 都 resize 到 h, w) ...
    if binary_pred_mask.shape != (h, w):
        binary_pred_mask = cv2.resize(binary_pred_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    if true_mask.shape != (h, w):
        true_mask = cv2.resize(true_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    if multiclass_pred_mask.shape != (h, w):
        multiclass_pred_mask = cv2.resize(multiclass_pred_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
    # --- 为二元遮罩上色 (保持不变) ---
    FG_COLOR_RGB = [255, 255, 255]
    true_colored = np.zeros((h, w, 3), dtype=np.uint8)
    binary_pred_colored = np.zeros((h, w, 3), dtype=np.uint8)
    true_colored[true_mask == 1] = FG_COLOR_RGB
    binary_pred_colored[binary_pred_mask == 1] = FG_COLOR_RGB

    # --- 核心修改點：為多類別遮罩上色，使用高對比度 Colormap ---
    
    # 1. 獲取唯一的類別 ID (不包括背景 0)
    unique_labels = np.unique(multiclass_pred_mask)
    unique_labels = unique_labels[unique_labels != 0]
    
    # 2. 選擇一個高對比度的 Colormap
    colormap = cm.get_cmap('tab20', len(unique_labels) if len(unique_labels) > 0 else 1)
    
    # 3. 手動創建彩色圖像
    multiclass_colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 創建一個從標籤ID到顏色的映射
    color_map = {label: (np.array(colormap(i)[:3]) * 255).astype(np.uint8) 
                 for i, label in enumerate(unique_labels)}

    for label_id, color_val in color_map.items():
        multiclass_colored[multiclass_pred_mask == label_id] = color_val

    # (可選) 打印每個類別的像素百分比，以檢查聚類坍塌
    print("\n--- Multiclass Prediction Stats ---")
    total_pixels = multiclass_pred_mask.size
    for label_id in unique_labels:
        pixel_count = np.sum(multiclass_pred_mask == label_id)
        percentage = (pixel_count / total_pixels) * 100
        print(f"  Class {label_id}: {pixel_count} pixels ({percentage:.2f}%)")
    print("------------------------------------")
        
    # --- 添加標題和拼接 (保持不變) ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    def add_title(image, text):
        img_with_title = cv2.copyMakeBorder(image, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.putText(img_with_title, text, (10, 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        return img_with_title

    img1 = add_title(cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR), "Original")
    img2 = add_title(cv2.cvtColor(true_colored, cv2.COLOR_RGB2BGR), "Ground Truth")
    img3 = add_title(cv2.cvtColor(binary_pred_colored, cv2.COLOR_RGB2BGR), "Binary Pred")
    img4 = add_title(cv2.cvtColor(multiclass_colored, cv2.COLOR_RGB2BGR), "Multiclass Pred")

    combined_image = np.concatenate((img1, img2, img3, img4), axis=1)
    
    cv2.imwrite(output_path, combined_image)


@ex.automain
def main(_run, _config, _log):
    # --- 環境設定 ---
    cudnn.enabled = True
    cudnn.benchmark = True
    device = torch.device(_config['device'])
    random.seed(_config['seed'])
    np.random.seed(_config['seed'])
    torch.manual_seed(_config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_config['seed'])

    # --- 1. 載入訓練好的模型 ---
    _log.info(f'###### 正在從 {_config["reload_model_path"]} 載入模型 ######')
    model = PrototypeSegmenter(cfg=_config['model_cfg']).to(device)
    model.load_state_dict(torch.load(_config['reload_model_path'], map_location=device, weights_only=True), strict=True)
    model.eval()

    # --- 2. 建立驗證資料集 ---
    _log.info('###### 正在載入驗證資料 ######')
    val_dataset = SupervisedPolypDataset(
        image_dir=_config['val_image_dir'],
        mask_dir=_config['val_mask_dir'],
        target_size=(_config['image_size'], _config['image_size']),
        return_original=True
    )
    val_loader = DataLoader(val_dataset, batch_size=_config.get('val_batch_size', 1), shuffle=False, num_workers=_config.get('num_workers', 4))

    # --- 3. 準備評估 ---
    metric = SegmentationMetric(num_classes=2)
    
    # 設定視覺化儲存
    num_images_to_save = 10
    output_dir = os.path.join(_run.observers[0].dir, 'segmentation_outputs') if _run.observers else './segmentation_outputs'
    os.makedirs(output_dir, exist_ok=True)
    indices_to_save = random.sample(range(len(val_dataset)), k=min(num_images_to_save, len(val_dataset)))
    _log.info(f"將為索引為 {indices_to_save} 的圖像儲存視覺化結果。")

    # --- 4. 驗證迴圈 ---
    _log.info('###### 開始標準分割驗證 ######')
    with torch.no_grad():
        for i, (images, masks, _, original_images, paths) in enumerate(tqdm(val_loader, desc="正在驗證")):
            images = images.to(device)
            
            # --- 模型前向傳播 ---
            # 確保模型在 eval 模式下返回兩個值
            logits, multiclass_map = model(images)

            # --- 處理二元預測 ---
            binary_pred_masks = logits.argmax(dim=1).cpu() # (B, H, W)
            
            # --- 處理多類別預測 ---
            multiclass_pred_masks = multiclass_map.argmax(dim=1).cpu() # (B, H, W)
            
            # 更新指標 (只用二元預測)
            metric.update(masks, binary_pred_masks)

            # --- 檢查是否需要儲存視覺化 ---
            for j in range(images.shape[0]):
                current_index = i * _config.get('val_batch_size', 1) + j
                if current_index in indices_to_save:
                    image_basename = os.path.basename(paths[j])
                    output_path = os.path.join(output_dir, f"result_{image_basename}")
                    
                    save_segmentation_results(
                        original_image=original_images[j].numpy(),
                        true_mask=masks[j].numpy(),
                        binary_pred_mask=binary_pred_masks[j].numpy(),
                        multiclass_pred_mask=multiclass_pred_masks[j].numpy(),
                        output_path=output_path
                    )
                    _log.info(f"已為索引 {current_index} 儲存分割結果。")

    # --- 5. 計算並打印最終結果 ---
    scores = metric.get_scores()
    # Class_Dice 是一個數組, [Dice_for_class_0, Dice_for_class_1]
    # 我們關心的是前景 (息肉)，即 class 1 的 Dice
    foreground_dice = scores["Class_Dice"][1]
    
    _log.info('驗證完成。')
    _log.info(f'標準驗證的平均 Dice 分數 (前景類別): {foreground_dice:.4f}')
    _log.info(f'平均 IoU (mIoU): {scores["Mean_IoU"]:.4f}')
    _run.log_scalar('validation_dice', foreground_dice)
    _run.log_scalar('validation_miou', scores["Mean_IoU"])