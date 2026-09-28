# 檔案: train_stage3.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models.MetricSegmenter import MetricSegmenter
from dataloaders.FinalTrainingDataset import FinalTrainingDataset
from dataloaders.SupervisedPolypDataset import SupervisedPolypDataset as ValidationDataset
from util.losses import CombinedSegLoss, PrototypeMetricLoss
from util.metric import SegmentationMetric

# --- 驗證函數 ---
def evaluate(model, dataloader, device):
    model.eval()
    metric = SegmentationMetric(num_classes=2)
    with torch.no_grad():
        # --- 核心修改點：正确解包 DataLoader 的返回值 ---
        # 即使 Dataset 返回了多个值，我们也只取前两个
        for data_batch in tqdm(dataloader, desc="正在驗證..."):
            # 手動解包，忽略我們不需要的值
            images, true_binary_masks = data_batch[0], data_batch[1]
            images, true_binary_masks = images.to(device), true_binary_masks.to(device)
            
            # 獲取模型預測
            # 模型在 eval 模式下返回 (logits, similarity_map)
            # 我們只關心 logits
            logits, _ = model(images)
            pred_binary = logits.argmax(dim=1)
            
            # 更新指標
            metric.update(true_binary_masks.cpu(), pred_binary.cpu())
            
    scores = metric.get_scores()
    foreground_dice = scores["Class_Dice"][1]
    return foreground_dice

# --- 主訓練流程 ---
def train_stage3(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 數據
    train_dataset = FinalTrainingDataset(image_dir=cfg['train_image_dir'], 
                                         binary_mask_dir=cfg['train_mask_dir'],
                                         sublabel_dir=cfg['sublabel_dir'])
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4)
    
    val_dataset = ValidationDataset(
        image_dir=cfg['val_image_dir'], 
        mask_dir=cfg['val_mask_dir'],
        return_original=False
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg['val_batch_size'], shuffle=False, num_workers=4)

    # 2. 模型
    model = MetricSegmenter(cfg=cfg['model_cfg']).to(device)
    
    # 3. 損失函數與優化器
    seg_loss_fn = CombinedSegLoss()
    metric_loss_fn = PrototypeMetricLoss(
        n_fg_clusters=cfg['model_cfg']['num_fg_prototypes']
    )
    metric_loss_weight = cfg['metric_loss_weight']
    
    # 分層學習率
    params = [
        {'params': model.encoder.parameters(), 'lr': cfg['encoder_lr']},
        {'params': model.decoder.parameters(), 'lr': cfg['decoder_lr']},
        {'params': model.fg_prototypes, 'lr': cfg['prototype_lr']},
        {'params': model.segmentation_head.parameters(), 'lr': cfg['head_lr']}
    ]
    optimizer = AdamW(params, weight_decay=cfg['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    # 4. 訓練與驗證迴圈
    best_val_dice = -1.0
    for epoch in range(cfg['epochs']):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        for images, binary_masks, sublabel_masks in progress_bar:
            images, binary_masks, sublabel_masks = images.to(device), binary_masks.to(device), sublabel_masks.to(device)
            
            optimizer.zero_grad()
            
            binary_logits, fg_similarity_map = model(images)
            
            # 計算兩種損失
            loss_seg = seg_loss_fn(binary_logits, binary_masks)
            
            sublabel_small = F.interpolate(sublabel_masks.unsqueeze(1).float(), 
                                           size=fg_similarity_map.shape[-2:],
                                           mode='nearest').squeeze(1).long()
            loss_metric = metric_loss_fn(fg_similarity_map, sublabel_small)
            
            total_loss = loss_seg + metric_loss_weight * loss_metric
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            progress_bar.set_postfix(loss=f'{total_loss.item():.4f}', seg_loss=f'{loss_seg.item():.4f}', metric_loss=f'{loss_metric.item():.4f}')
        
        scheduler.step()
        
        # 運行驗證
        current_val_dice = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} - 訓練損失: {epoch_loss/len(train_loader):.4f}, 驗證 Dice: {current_val_dice:.4f}")
        
        # 保存最佳模型
        if current_val_dice > best_val_dice:
            best_val_dice = current_val_dice
            os.makedirs(cfg['output_dir'], exist_ok=True)
            save_path = os.path.join(cfg['output_dir'], "best_model_stage3_mlw005.pth")
            torch.save(model.state_dict(), save_path)
            print(f"****** 新的最佳模型！ Dice: {best_val_dice:.4f}。已儲存至: {save_path} ******")
            
if __name__ == '__main__':
    config = {
        # 數據路徑
        'train_image_dir': '/home/U116med/data/polyp/TrainDataset/images/',
        'train_mask_dir': '/home/U116med/data/polyp/TrainDataset/masks/',
        'sublabel_dir': './polypdata/sublabels_consistent_k8/',
        'val_image_dir': '/home/U116med/data/polyp/TestDataset/test/images/',
        'val_mask_dir': '/home/U116med/data/polyp/TestDataset/test/masks/',
        
        'image_size': 352,
        'batch_size': 4,
        'val_batch_size': 4,
        'epochs': 100,
        
        # 模型配置
        'model_cfg': {
            'stage2_pretrained_path': './pretrained_stage2_weighted/stage2_epoch_20.pth',
            'decoder_out_channels': 256,
            'num_fg_prototypes': 4, # 與子標籤生成時的前景類別數一致
        },
        
        # 優化器
        'encoder_lr': 1e-6,
        'decoder_lr': 1e-5,
        'head_lr': 1e-4,
        'prototype_lr': 1e-4, # 原型也需要微調
        'weight_decay': 1e-2,

        # 損失
        'metric_loss_weight': 0.05, # 輔助損失的權重通常設得較小
        
        # 輸出
        'output_dir': './final_model_metric/',
    }
    
    train_stage3(config)