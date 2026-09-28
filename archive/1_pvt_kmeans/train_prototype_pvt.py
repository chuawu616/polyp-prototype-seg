import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from config import ex
from models.PrototypeSegmenter import PrototypeSegmenter
from dataloaders.SupervisedPolypDataset import SupervisedPolypDataset
from util.metric import SegmentationMetric

class CombinedLoss(nn.Module):
    """
    一個組合了 BCE Loss 和 IoU Loss 的損失函數。
    """
    def __init__(self, bce_weight=0.5, iou_weight=0.5, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, true_mask):
        bce = self.bce_loss(logits[:, 1, :, :], true_mask.float())

        probs = torch.sigmoid(logits[:, 1, :, :])
        probs_flat = probs.view(probs.shape[0], -1)
        mask_flat = true_mask.view(true_mask.shape[0], -1).float()
        
        intersection = (probs_flat * mask_flat).sum(1)
        union = (probs_flat + mask_flat).sum(1) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        iou_loss = 1 - iou.mean()

        total_loss = self.bce_weight * bce + self.iou_weight * iou_loss
        return total_loss
        
def evaluate(model, dataloader, device, cfg):
    """
    在驗證集上評估模型性能。

    Returns:
        float: 前景類別的平均 Dice 分數。
    """
    model.eval()  # 將模型設置為評估模式
    metric = SegmentationMetric(num_classes=2)
    
    with torch.no_grad():
        for images, masks, prototypes in tqdm(dataloader, desc="正在驗證..."):
            images = images.to(device)
            prototypes = prototypes.to(device)
            
            # 模型前向傳播
            logits = model(images, per_image_prototypes=prototypes)
            pred_masks = logits.argmax(dim=1).cpu() # (B, H, W)
            
            # 更新指標
            metric.update(masks, pred_masks)
            
    scores = metric.get_scores()
    foreground_dice = scores["Class_Dice"][1]
    return foreground_dice

# --- 主訓練函式 ---
# 使用 @ex.automain 裝飾器來標記這個函數為 Sacred 實驗的入口點
@ex.automain
def main(_run, _config, _log):
    # --- 1. 準備資料 ---
    _log.info("正在準備資料集...")
    dataset = SupervisedPolypDataset(
        image_dir=_config['train_image_dir'], # 確保 config 中有 train_image_dir
        mask_dir=_config['train_mask_dir'],   # 確保 config 中有 train_mask_dir
        prototype_dir=_config['train_prototype_dir'],
        target_size=(_config['image_size'], _config['image_size'])
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=_config['batch_size'], 
        shuffle=True, 
        num_workers=_config.get('num_workers', 4), # 使用 .get 增加靈活性
        pin_memory=True,
        drop_last=True
    )
    
    val_dataset = SupervisedPolypDataset(
        image_dir=_config['val_image_dir'],
        mask_dir=_config['val_mask_dir'],
        prototype_dir=_config['val_prototype_dir'], # 驗證集也需要自己的原型
        target_size=(_config['image_size'], _config['image_size'])
    )
    val_loader = DataLoader(val_dataset, batch_size=_config.get('val_batch_size', 1), shuffle=False,
                            num_workers=_config.get('num_workers', 4))
    
    # --- 2. 準備模型 ---
    _log.info("正在建立模型...")
    model = PrototypeSegmenter(cfg=_config['model_cfg']).to(_config['device'])
    
    # --- 3. 準備損失函數和優化器 ---
    _log.info("正在設定損失函數與優化器...")
    loss_cfg = _config.get('loss_cfg', {'bce_weight': 0.5, 'iou_weight': 0.5}) # 提供預設值
    criterion = CombinedLoss(bce_weight=loss_cfg['bce_weight'], iou_weight=loss_cfg['iou_weight']).to(_config['device'])
    
    # 設定分層學習率
    optimizer_cfg = _config['optimizer_cfg']
    encoder_params = [p for n, p in model.named_parameters() if 'encoder.' in n and p.requires_grad]
    decoder_params = [p for n, p in model.named_parameters() if 'decoder.' in n and p.requires_grad]

    params_to_optimize = [
        {'params': encoder_params, 'lr': optimizer_cfg['encoder_lr']},
        {'params': decoder_params, 'lr': optimizer_cfg['decoder_lr']},
    ]
    
    optimizer = torch.optim.AdamW(params_to_optimize, weight_decay=optimizer_cfg['weight_decay'])
    
    # 設定學習率調度器
    scheduler = CosineAnnealingLR(optimizer, T_max=_config['epochs'], eta_min=1e-6)

    # --- 4. 訓練迴圈 ---
    _log.info("###### 開始訓練與驗證 ######")

    best_val_dice = -1.0
    best_epoch = -1
    
    # 獲取快照保存路徑
    snapshot_dir = os.path.join(_run.observers[0].dir, 'snapshots') if _run.observers else './snapshots'
    os.makedirs(snapshot_dir, exist_ok=True)
    
    for epoch in range(_config['epochs']):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{_config['epochs']}")
        for images, masks, prototypes in progress_bar:
            images = images.to(_config['device'])
            masks = masks.to(_config['device'])
            prototypes = prototypes.to(_config['device'])
            
            optimizer.zero_grad()
            
            logits = model(images, per_image_prototypes=prototypes)
            loss = criterion(logits, masks)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f'{loss.item():.4f}')

        avg_epoch_loss = epoch_loss / len(dataloader)
        _log.info(f"Epoch {epoch+1} 完成, 平均損失: {avg_epoch_loss:.4f}, 當前學習率(Encoder): {optimizer.param_groups[0]['lr']:.6f}")
        _run.log_scalar("epoch_loss", avg_epoch_loss, epoch)

        # --- 驗證階段 ---
        # 每隔 N 個 epoch (或每個 epoch) 運行一次驗證
        
        if (epoch + 1) % _config.get('eval_every_epochs', 1) == 0:
            current_val_dice = evaluate(model, val_loader, _config['device'], _config)
            _log.info(f"Epoch {epoch+1} 完成 - 訓練損失: {avg_epoch_loss:.4f}, 驗證 Dice: {current_val_dice:.4f}")
            _run.log_scalar("val_dice", current_val_dice, epoch)
            
            if current_val_dice > best_val_dice:
                best_val_dice = current_val_dice
                best_epoch = epoch + 1
                
                # 定義最佳模型的文件路徑
                best_model_path = os.path.join(snapshot_dir, "best_model.pth")
                
                # 保存模型
                torch.save(model.state_dict(), best_model_path)
                _log.info(f"****** 新的最佳模型 Dice: {best_val_dice:.4f}。已儲存至: {best_model_path} ******")
        
        scheduler.step()
        

    _log.info("###### 訓練完成 ######")
    _log.info(f"最佳模型出現在 Epoch {best_epoch}，其驗證 Dice 分數為: {best_val_dice:.4f}")