# 檔案: train_baseline.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# 導入模型和數據
from models.BaselineSegmenter import BaselineSegmenter
# 我們可以使用之前強大的 FinalTrainingDataset (帶 Albumentations)，但只取前兩個返回值
from dataloaders.FinalTrainingDataset import FinalTrainingDataset 
from dataloaders.SupervisedPolypDataset import SupervisedPolypDataset as ValidationDataset
from util.losses_structure import StructureLoss
from util.metric import SegmentationMetric

def validation_collate(batch):
    """
    只保留 image 和 mask，丟棄原始圖像和路徑以避免 stack 錯誤。
    batch: list of tuples (image, mask, original, path)
    """
    images = [item[0] for item in batch]
    masks = [item[1] for item in batch]
    
    # 將 list of tensors 堆疊成 batch tensor
    images = torch.stack(images, 0)
    masks = torch.stack(masks, 0)
    
    return images, masks

def evaluate(model, dataloader, device):
    model.eval()
    metric = SegmentationMetric(num_classes=2)
    with torch.no_grad():
        # --- 修改點：現在 dataloader 只返回兩個值 ---
        for images, true_binary_masks in tqdm(dataloader, desc="驗證中...", leave=False):
            images, true_binary_masks = images.to(device), true_binary_masks.to(device)
            
            logits = model(images)
            pred_binary = (logits > 0).long().squeeze(1)
            
            metric.update(true_binary_masks.cpu(), pred_binary.cpu())
    scores = metric.get_scores()
    return scores["Class_Dice"][1]

def train_baseline(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 數據
    # 這裡我們需要稍微修改 Dataset 的使用方式，因為 FinalTrainingDataset 返回 3 個值
    # 我們在循環中只取前兩個
    train_dataset = FinalTrainingDataset(
        image_dir=cfg['train_image_dir'], 
        binary_mask_dir=cfg['train_mask_dir'],
        sublabel_dir=cfg['train_mask_dir'], # 這裡隨便填一個存在的目錄即可，因為我們不用它
        image_size=(352, 352)
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    
    val_dataset = ValidationDataset(image_dir=cfg['val_image_dir'], mask_dir=cfg['val_mask_dir'], return_original=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['val_batch_size'], shuffle=False, num_workers=4, collate_fn=validation_collate)

    # 2. 模型
    print("###### 初始化 Baseline 模型 (PVTv2 + EMCAD) ######")
    model = BaselineSegmenter(pretrained=True).to(device)
    
    # 3. 損失函數
    # 您的 StructureLoss 是為 (B, 2, H, W) 設計的，我們需要一個適配 (B, 1, H, W) 的版本
    # 或者簡單地修改模型輸出為 2 通道。為了簡單，我們在這裡做一個適配器。
    
    class BinaryStructureLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.loss_fn = StructureLoss()
        def forward(self, pred_logit, mask):
            # pred_logit: (B, 1, H, W)
            # 構造成 (B, 2, H, W): channel 0 是 -logit, channel 1 是 logit
            # 這樣 softmax 後就等效於 sigmoid
            pred_2ch = torch.cat([-pred_logit, pred_logit], dim=1)
            return self.loss_fn(pred_2ch, mask)

    criterion = BinaryStructureLoss()

    # 4. 優化器
    # 這裡我們可以用一個統一的學習率，或者繼續分層
    optimizer = AdamW([
        {'params': model.encoder.parameters(), 'lr': cfg['encoder_lr']},
        {'params': model.decoder.parameters(), 'lr': cfg['decoder_lr']},
        {'params': model.head.parameters(), 'lr': cfg['decoder_lr']}
    ], weight_decay=cfg['weight_decay'])
    
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=1e-6)

    # 5. 訓練循環
    print("###### 開始訓練 ######")
    best_val_dice = -1.0
    
    for epoch in range(cfg['epochs']):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        
        # Ramp-up 不再需要，因為沒有輔助損失
        
        for batch in pbar:
            # 解包 (image, mask, sublabel) -> 只取前兩個
            images = batch[0].to(device)
            masks = batch[1].to(device)
            
            optimizer.zero_grad()
            
            logits = model(images) # (B, 1, H, W)
            loss = criterion(logits, masks)
            
            loss.backward()
            # 梯度剪裁
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        scheduler.step()
        
        # 驗證與保存
        current_val_dice = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} - Train Loss: {epoch_loss/len(train_loader):.4f}, Val Dice: {current_val_dice:.4f}")
        
        if current_val_dice > best_val_dice:
            best_val_dice = current_val_dice
            os.makedirs(cfg['output_dir'], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(cfg['output_dir'], "best_model_baseline.pth"))
            print(f"****** 新最佳模型 (Dice {best_val_dice:.4f}) 已保存 ******")

if __name__ == '__main__':
    config = {
        'train_image_dir': '/home/U116med/data/polyp/TrainDataset/images/',
        'train_mask_dir': '/home/U116med/data/polyp/TrainDataset/masks/',
        'val_image_dir': '/home/U116med/data/polyp/TestDataset/test/images/',
        'val_mask_dir': '/home/U116med/data/polyp/TestDataset/test/masks/',
        
        'batch_size': 4, # 如果 OOM，改為 2
        'val_batch_size': 4,
        'epochs': 100,
        
        # 學習率：Baseline 通常可以用稍微大一點的 LR
        'encoder_lr': 1e-4, 
        'decoder_lr': 1e-4,
        'weight_decay': 1e-2,
        
        'output_dir': './final_model_baseline/',
    }
    
    train_baseline(config)