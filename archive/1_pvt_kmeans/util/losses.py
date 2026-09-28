# 檔案: util/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedSegLoss(nn.Module):
    """
    組合了 BCE Loss 和 Dice Loss 的分割損失。
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super(CombinedSegLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, true_binary_mask):
        # BCE Loss
        # 只對前景 logit 計算 (index 1)
        bce = self.bce_loss(logits[:, 1, :, :], true_binary_mask.float())

        # Dice Loss
        probs = torch.sigmoid(logits[:, 1, :, :])
        probs_flat = probs.view(probs.shape[0], -1)
        mask_flat = true_binary_mask.view(true_binary_mask.shape[0], -1).float()
        
        intersection = (probs_flat * mask_flat).sum(1)
        dice_score = (2. * intersection + self.smooth) / (probs_flat.sum(1) + mask_flat.sum(1) + self.smooth)
        dice_loss = 1 - dice_score.mean()
        
        return self.bce_weight * bce + self.dice_weight * dice_loss

class PrototypeMetricLoss(nn.Module):
    """
    原型度量損失，一個有監督的對比損失。
    """
    def __init__(self, temperature=0.1, n_fg_clusters=4):
        super().__init__()
        self.temperature = temperature
        self.n_fg_clusters = n_fg_clusters # 明確知道前景子類的數量
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, similarity_map, sublabels_small):
        # similarity_map: (B, K_fg, H', W'), K_fg 就是 n_fg_clusters
        # sublabels_small: (B, H', W'), 值為 1-8
        
        # 前景子標簽的值範圍是 [1, n_fg_clusters]
        fg_mask = (sublabels_small > 0) & (sublabels_small <= self.n_fg_clusters)
        
        if not fg_mask.any():
            # 如果這個批次中沒有任何前景像素，則損失為0
            return torch.tensor(0.0, device=similarity_map.device)
            
        # 將 sublabels 轉換為 0-indexed (0 to n_fg_clusters-1)
        # 我們只取前景區域的標簽
        target = sublabels_small[fg_mask] - 1
        
        # 獲取對應前景像素的相似度分數
        # similarity_map.permute(0, 2, 3, 1) -> (B, H', W', K_fg)
        # [fg_mask] -> (N_fg_pixels, K_fg)
        logits = similarity_map.permute(0, 2, 3, 1)[fg_mask] / self.temperature
        
        # 現在，logits 的第二維度是 K_fg，target 的值範圍是 [0, K_fg-1]
        # 維度完全匹配，斷言不會失敗
        return self.criterion(logits, target)