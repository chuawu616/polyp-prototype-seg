# 檔案: train_stage3_emcad.py
# 描述: 使用 EMCAD (CASCADE) Decoder 的第三階段訓練腳本。

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from models.PrototypeSegmenter_EMCAD import PrototypeSegmenterEMCAD

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
            
            # 模型返回 (binary_logits, k_class_logits)
            logits_binary, _ = model(images) 
            
            pred_binary = logits_binary.argmax(dim=1)
            metric.update(true_binary_masks.cpu(), pred_binary.cpu())
            
    scores = metric.get_scores()
    return scores["Class_Dice"][1]

def train_stage3_emcad(cfg):
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
    # 使用新的 EMCAD 模型
    model = PrototypeSegmenterEMCAD(cfg=cfg['model_cfg']).to(device)
    
    # 3. 損失函數
    structure_loss_fn = StructureLoss()
    sublabel_loss_fn = SublabelLoss()
    reg_loss_fn = RegularizationLoss(w_scale=0.5, w_iron=0.01) 
    aux_weight = cfg['aux_loss_weight']

    # 4. 優化器
    # EMCAD Decoder 的參數比 FPN 多，可能需要微調學習率
    optimizer = AdamW([
        {'params': model.encoder.parameters(), 'lr': cfg['encoder_lr']},
        {'params': model.decoder.parameters(), 'lr': cfg['decoder_lr']},
        {'params': model.prototypes, 'lr': cfg['prototype_lr']}
    ], weight_decay=cfg['weight_decay'])
    
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    # 5. 訓練循環
    print("###### 開始第三階段 (EMCAD Decoder + Pure Prototype) 訓練 ######")
    best_val_dice = -1.0
    
    for epoch in range(cfg['epochs']):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        
        # Ramp-up logic
        if epoch < 5:
            ramp_weight = 0.0
        else:
            ramp_weight = min(1.0, (epoch - 5) / 10.0)
        
        for images, binary_masks, sublabel_masks in pbar:
            images = images.to(device)
            binary_masks = binary_masks.to(device)
            sublabel_masks = sublabel_masks.to(device)
            
            optimizer.zero_grad()
            
            # 前向傳播
            logits_binary, logits_k_class = model(images)
            
            # 計算主要損失
            loss_main = structure_loss_fn(logits_binary, binary_masks)
            
            # 計算輔助損失
            # 注意：PrototypeSegmenterEMCAD 的 logits_k_class 已經上採樣了
            # 所以 sublabel_masks 不需要 resize
            loss_aux = sublabel_loss_fn(logits_k_class, sublabel_masks)

            # 計算正則化損失
            loss_reg = reg_loss_fn(logits_k_class)
            
            total_loss = loss_main + aux_weight * loss_aux + ramp_weight * loss_reg
            
            total_loss.backward()
            
            # 強烈建議：加入梯度剪裁，因為 EMCAD 結構較深，容易出現梯度不穩
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += total_loss.item()
            pbar.set_postfix(loss=f"{total_loss.item():.4f}", 
                             main=f"{loss_main.item():.4f}", 
                             aux=f"{loss_aux.item():.4f}")
        
        scheduler.step()
        
        # 驗證與保存
        current_val_dice = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} - Train Loss: {epoch_loss/len(train_loader):.4f}, Val Dice: {current_val_dice:.4f}")
        
        if current_val_dice > best_val_dice:
            best_val_dice = current_val_dice
            os.makedirs(cfg['output_dir'], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(cfg['output_dir'], "best_model_emcad.pth"))
            print(f"****** 新最佳模型 (Dice {best_val_dice:.4f}) 已保存 ******")

if __name__ == '__main__':
    config = {
        'train_image_dir': '/home/U116med/data/polyp/TrainDataset/images/',
        'train_mask_dir': '/home/U116med/data/polyp/TrainDataset/masks/',
        'sublabel_dir': './polypdata/sublabels_consistent_k8/',
        'val_image_dir': '/home/U116med/data/polyp/TestDataset/test/images/',
        'val_mask_dir': '/home/U116med/data/polyp/TestDataset/test/masks/',
        
        'batch_size': 4, # 注意：EMCAD 可能比 FPN 更佔顯存，如果 OOM 請降為 2
        'val_batch_size': 4,
        'epochs': 100,
        
        'model_cfg': {
            # 使用您之前訓練好的 Encoder 權重
            'stage2_pretrained_path': './pretrained_stage2_weighted/stage2_epoch_20.pth',
            # 這裡不需要 decoder_out_channels，因為 CASCADE 的輸出通道由 PVT Stage 1 決定 (64)
            # 除非您在 PrototypeSegmenter_EMCAD 中加了額外的 conv
            'num_fg_prototypes': 4,
            'num_bg_prototypes': 4, 
        },
        
        'encoder_lr': 1e-4,
        'decoder_lr': 1e-4,
        'prototype_lr': 1e-3,
        'weight_decay': 1e-2,
        
        'aux_loss_weight': 0.3, 
        'output_dir': './final_model_emcad_enlr-4/', # 新的輸出目錄
    }
    
    train_stage3_emcad(config)