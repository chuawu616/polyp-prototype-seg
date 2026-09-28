# 檔案: train_stage2.py
# 描述: 執行有監督的像素級對比學習的第二階段微調 (带有前景-背景负样本加权)。

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models.PixelContrastiveModel import PixelContrastiveModel
from dataloaders.SupervisedContrastiveDataset import SupervisedContrastiveDataset

class SupervisedPixelContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, n_fg_clusters=4, base_negative_weight=1.0, fg_bg_neg_weight=2.0):
        super(SupervisedPixelContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.n_fg_clusters = n_fg_clusters
        self.base_negative_weight = base_negative_weight
        self.fg_bg_neg_weight = fg_bg_neg_weight

    def forward(self, features, labels):
        """
        Args:
            features (Tensor): 投影後的特徵圖 z (B, C, H, W)。
            labels (Tensor): 子標籤圖 (B, H, W)。
        """
        B, C, H, W = features.shape
        N = H * W
        
        # 準備特徵和標籤
        features_flat = F.normalize(features, p=2, dim=1).view(B, C, -1).permute(0, 2, 1) # (B, N, C)
        labels_flat = labels.view(B, -1) # (B, N)
        
        # 1. 構建正樣本對的 Mask
        mask_positive_pairs = (labels_flat.unsqueeze(1) == labels_flat.unsqueeze(2)) & (labels_flat.unsqueeze(1) > 0)
        identity_mask = torch.eye(N, dtype=torch.bool, device=features.device).unsqueeze(0)
        mask_positive_pairs = mask_positive_pairs & ~identity_mask

        # 2. 計算相似度矩陣 (logits)
        logits = torch.div(torch.matmul(features_flat, features_flat.transpose(1, 2)), self.temperature)
        
        # 3. 應用負樣本權重 (如果需要)
        if self.base_negative_weight != 1.0 or self.fg_bg_neg_weight != 1.0:
            is_fg = (labels_flat > 0) & (labels_flat <= self.n_fg_clusters)
            is_bg = labels_flat > self.n_fg_clusters
            mask_fg_vs_bg_negs = (is_fg.unsqueeze(2) & is_bg.unsqueeze(1)) | \
                                   (is_bg.unsqueeze(2) & is_fg.unsqueeze(1))
            
            weights = torch.full_like(logits, self.base_negative_weight)
            weights[mask_fg_vs_bg_negs] = self.fg_bg_neg_weight
            weights[mask_positive_pairs] = 1.0
            weights.diagonal(dim1=-2, dim2=-1).fill_(1.0)
            
            logits += torch.log(weights)
        
        # 4. **核心修正**: 計算 InfoNCE 損失的標準方法
        
        # a. 為了數值穩定性，從每行減去最大值
        logits_max, _ = torch.max(logits, dim=2, keepdim=True)
        logits = logits - logits_max.detach()

        # b. 計算分母：exp(logits) 的總和 (排除自身)
        #    我們不使用 masked_fill(-inf)，而是直接在求和時排除對角線
        exp_logits = torch.exp(logits)
        exp_logits = exp_logits.clone() # 創建一個副本以進行原地修改
        exp_logits.diagonal(dim1=-2, dim2=-1).fill_(0) # 將對角線設為0
        log_prob_denominator = torch.log(exp_logits.sum(2) + 1e-8)

        # c. 計算分子：正樣本對的 log_prob
        #    我們只關心正樣本對的 logits
        #    將非正樣本對的位置設為一個極小值，這樣它們在 logsumexp 中就不會被考慮
        log_prob_numerator = logits.clone() # 創建副本
        log_prob_numerator[~mask_positive_pairs] = -1e9
        
        # 使用 logsumexp 技巧來穩定地計算 log(sum(exp(positives)))
        # 這等效於對所有正樣本 logits 求和的對數
        log_prob_numerator = torch.logsumexp(log_prob_numerator, dim=2)
        
        # d. 計算 InfoNCE 損失
        #    Loss = - (分子 - 分母)
        #    我們只對有正樣本的 anchor (行) 計算損失
        num_positives = mask_positive_pairs.sum(2)
        has_positives_mask = num_positives > 0
        
        loss = - (log_prob_numerator[has_positives_mask] - log_prob_denominator[has_positives_mask])
        
        # e. 對損失進行歸一化
        #    原始 InfoNCE 是對每個 anchor 的所有正樣本求平均，我們也這樣做
        loss = loss / num_positives[has_positives_mask]
        
        # 最終損失是所有有效 anchor 損失的平均值
        final_loss = loss.mean()

        # 處理整個 batch 都沒有正樣本的極端情況
        if torch.isnan(final_loss):
            return torch.tensor(0.0, device=features.device)

        return final_loss

# --- 主訓練流程 (與之前版本相同) ---
def train_stage2(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = SupervisedContrastiveDataset(image_dir=cfg['train_image_dir'], sublabel_dir=cfg['sublabel_dir'])
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4)
    
    model = PixelContrastiveModel(
        pretrained=False, 
        target_stages=cfg['target_stages'],
        proj_dim=cfg['proj_dim']
    ).to(device)
    
    state_dict = torch.load(cfg['stage1_pretrained_path'], map_location=device)
    model.encoder.load_state_dict(state_dict)
    print(f"成功從 {cfg['stage1_pretrained_path']} 加載第一階段權重！")
    
    # 實例化新的損失函數
    criterion = SupervisedPixelContrastiveLoss(
        temperature=cfg['temperature'],
        n_fg_clusters=cfg['n_fg_clusters'],
        base_negative_weight=cfg['base_negative_weight'], 
        fg_bg_neg_weight=cfg['fg_bg_neg_weight']
    )
    optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    print("###### 開始第二階段 (帶權重監督對比) 微調 ######")
    for epoch in range(cfg['epochs']):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        for view1, sublabel1, view2, sublabel2 in progress_bar:
            images = torch.cat([view1, view2], dim=0).to(device)
            sublabels = torch.cat([sublabel1, sublabel2], dim=0).to(device)

            optimizer.zero_grad()
            projections_list = model(images)
            
            total_loss = 0
            for z in projections_list:
                sublabel_small = F.interpolate(sublabels.unsqueeze(1).float(), 
                                               size=z.shape[-2:], 
                                               mode='nearest').squeeze(1).long()
                total_loss += criterion(z, sublabel_small)

            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            progress_bar.set_postfix(loss=f'{total_loss.item():.4f}')
        
        scheduler.step()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} 完成, 平均損失: {avg_epoch_loss:.4f}")

        if (epoch + 1) % cfg['save_every_epochs'] == 0:
            os.makedirs(cfg['output_dir'], exist_ok=True)
            save_path = os.path.join(cfg['output_dir'], f"stage2_epoch_{epoch+1}.pth")
            torch.save(model.encoder.state_dict(), save_path)
            print(f"已儲存【Encoder】的權重快照至: {save_path}")

if __name__ == '__main__':
    # --- 配置參數 ---
    config = {
        # 數據路徑
        'train_image_dir': '/home/U116med/data/polyp/TrainDataset/images/',
        'sublabel_dir': './polypdata/sublabels_k8/',
        
        'stage1_pretrained_path': './pretrained_stage1_finetuned_on_test/stage1_epoch_20.pth',
        
        'batch_size': 4,
        'epochs': 20,
        'image_size': 352,
        
        # 模型
        'target_stages': [0, 1, 2, 3],
        'proj_dim': 128,
        
        # --- 損失函數的新增配置 ---
        'temperature': 0.1,
        'n_fg_clusters': 4, # 告訴損失函數前景子類別的數量 (1-4)
        'base_negative_weight': 1.0,
        'fg_bg_neg_weight': 2.0, # 讓前景-背景的負樣本損失權重是普通負樣本的2倍
        
        # 優化器
        'lr': 1e-5,
        'weight_decay': 1e-2,
        
        # 輸出
        'output_dir': './pretrained_stage2_weighted/', # 使用新目錄
        'save_every_epochs': 10,
    }
    
    train_stage2(config)