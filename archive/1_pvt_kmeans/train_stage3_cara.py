# 檔案: train_stage3_cara.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# --- 核心修改：導入 CARA 模型 ---
from models.PrototypeSegmenter_CARA import PrototypeSegmenterCARA

from dataloaders.FinalTrainingDataset import FinalTrainingDataset 
from dataloaders.SupervisedPolypDataset import SupervisedPolypDataset as ValidationDataset
from util.losses_reg import StructureLoss, SublabelLoss, RegularizationLoss
from util.metric import SegmentationMetric

def evaluate(model, dataloader, device):
    model.eval()
    metric = SegmentationMetric(num_classes=2)
    with torch.no_grad():
        for images, true_binary_masks, _ in tqdm(dataloader, desc="驗證中...", leave=False):
            images, true_binary_masks = images.to(device), true_binary_masks.to(device)
            logits_binary, _ = model(images) 
            pred_binary = logits_binary.argmax(dim=1)
            metric.update(true_binary_masks.cpu(), pred_binary.cpu())
    scores = metric.get_scores()
    return scores["Class_Dice"][1]

def train_stage3_cara(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 數據
    train_dataset = FinalTrainingDataset(
        image_dir=cfg['train_image_dir'], 
        binary_mask_dir=cfg['train_mask_dir'],
        sublabel_dir=cfg['sublabel_dir']
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4)
    
    val_dataset = ValidationDataset(image_dir=cfg['val_image_dir'], mask_dir=cfg['val_mask_dir'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['val_batch_size'], shuffle=False, num_workers=4)

    # 2. 模型
    model = PrototypeSegmenterCARA(cfg=cfg['model_cfg']).to(device)
    
    # 3. 損失函數
    structure_loss_fn = StructureLoss()
    sublabel_loss_fn = SublabelLoss()
    reg_loss_fn = RegularizationLoss(w_scale=0.5, w_iron=0.01) 
    aux_weight = cfg['aux_loss_weight']

    # 4. 優化器
    optimizer = AdamW([
        {'params': model.encoder.parameters(), 'lr': cfg['encoder_lr']},
        {'params': model.decoder.parameters(), 'lr': cfg['decoder_lr']},
        {'params': model.prototypes, 'lr': cfg['prototype_lr']}
    ], weight_decay=cfg['weight_decay'])
    
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    # 5. 訓練循環
    print("###### 開始第三階段 (CaraNet Decoder + Pure Prototype) 訓練 ######")
    best_val_dice = -1.0
    
    for epoch in range(cfg['epochs']):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        
        if epoch < 5:
            ramp_weight = 0.0
        else:
            ramp_weight = min(1.0, (epoch - 5) / 10.0)
        
        for images, binary_masks, sublabel_masks in pbar:
            images = images.to(device)
            binary_masks = binary_masks.to(device)
            sublabel_masks = sublabel_masks.to(device)
            
            optimizer.zero_grad()
            
            logits_binary, logits_k_class = model(images)
            
            loss_main = structure_loss_fn(logits_binary, binary_masks)
            loss_aux = sublabel_loss_fn(logits_k_class, sublabel_masks)
            loss_reg = reg_loss_fn(logits_k_class)
            
            total_loss = loss_main + aux_weight * loss_aux + ramp_weight * loss_reg
            
            total_loss.backward()
            # 梯度剪裁
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            pbar.set_postfix(loss=f"{total_loss.item():.4f}", main=f"{loss_main.item():.4f}")
        
        scheduler.step()
        
        current_val_dice = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} - Train Loss: {epoch_loss/len(train_loader):.4f}, Val Dice: {current_val_dice:.4f}")
        
        if current_val_dice > best_val_dice:
            best_val_dice = current_val_dice
            os.makedirs(cfg['output_dir'], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(cfg['output_dir'], "best_model_cara.pth"))
            print(f"****** 新最佳模型 (Dice {best_val_dice:.4f}) 已保存 ******")

if __name__ == '__main__':
    config = {
        'train_image_dir': '/home/U116med/data/polyp/TrainDataset/images/',
        'train_mask_dir': '/home/U116med/data/polyp/TrainDataset/masks/',
        'sublabel_dir': './polypdata/sublabels_consistent_k8/',
        'val_image_dir': '/home/U116med/data/polyp/TestDataset/test/images/',
        'val_mask_dir': '/home/U116med/data/polyp/TestDataset/test/masks/',
        
        'batch_size': 4, 
        'val_batch_size': 4,
        'epochs': 100,
        
        'model_cfg': {
            'stage2_pretrained_path': './pretrained_stage2_weighted/stage2_epoch_20.pth',
            # CaraNet Decoder 輸出通道設為 64
            'decoder_out_channels': 64, 
            'num_fg_prototypes': 4,
            'num_bg_prototypes': 4, 
        },
        
        'encoder_lr': 1e-4,
        'decoder_lr': 1e-4,
        'prototype_lr': 1e-3,
        'weight_decay': 1e-2,
        
        'aux_loss_weight': 0.3, 
        'output_dir': './final_model_cara_enlr-4/',
    }
    
    train_stage3_cara(config)