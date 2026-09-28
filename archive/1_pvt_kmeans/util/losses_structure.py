# 檔案: util/losses_structure.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class StructureLoss(nn.Module):
    """
    基於邊緣加權的結構損失，用於二元分割。
    """
    def __init__(self):
        super(StructureLoss, self).__init__()

    def forward(self, pred, mask):
        # pred: (B, 2, H, W), mask: (B, H, W)
        mask_float = mask.unsqueeze(1).float()
        
        wb = torch.ones_like(mask_float)
        wb = 1 + 5 * torch.abs(F.avg_pool2d(mask_float, kernel_size=31, stride=1, padding=15) - mask_float)

        wbce = F.cross_entropy(pred, mask, reduction='none').unsqueeze(1)
        wbce = (wb * wbce).sum(dim=(2, 3)) / wb.sum(dim=(2, 3))

        pred_prob = F.softmax(pred, dim=1)[:, 1:2, :, :]
        inter = ((pred_prob * mask_float) * wb).sum(dim=(2, 3))
        union = ((pred_prob + mask_float) * wb).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()

class SublabelLoss(nn.Module):
    def __init__(self):
        super(SublabelLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, logits_k_class, sublabel_map):
        # logits_k_class: (B, K, H, W)
        # sublabel_map: (B, H, W)
        
        num_classes = logits_k_class.shape[1]
        
        target = sublabel_map.clone().long() # 確保是 long 類型
        
        # 1. 處理 0 值 (背景/無效區域)
        # 原始 0 值應該被忽略
        ignore_mask = target == 0
        
        # 2. 轉換為 0-indexed
        # 所有 >0 的值減 1 (1->0, 8->7)
        target[target > 0] -= 1
        
        # 3. **關鍵安全檢查**: 過濾掉超出範圍的標籤
        # 如果 target >= num_classes，說明這個標籤對於當前模型來說是無效的
        # 這通常發生在配置不匹配時
        invalid_mask = target >= num_classes
        
        # 將所有無效區域 (原始為0 或 超出範圍) 設為 ignore_index (-100)
        target[ignore_mask | invalid_mask] = -100
        
        return self.ce(logits_k_class * 10.0, target)