import torch
import torch.nn as nn
import torch.nn.functional as F

class StructureLoss(nn.Module):
    """
    基於邊緣加權的結構損失 (Weighted BCE + Weighted IoU)，用於二元分割。
    """
    def __init__(self):
        super(StructureLoss, self).__init__()

    def forward(self, pred, mask):
        # pred: (B, 2, H, W), mask: (B, H, W)
        mask_float = mask.unsqueeze(1).float()
        
        # 計算邊緣權重
        wb = torch.ones_like(mask_float)
        wb = 1 + 5 * torch.abs(F.avg_pool2d(mask_float, kernel_size=31, stride=1, padding=15) - mask_float)

        # Weighted BCE
        wbce = F.cross_entropy(pred, mask, reduction='none').unsqueeze(1)
        wbce = (wb * wbce).sum(dim=(2, 3)) / wb.sum(dim=(2, 3))

        # Weighted IoU
        pred_prob = F.softmax(pred, dim=1)[:, 1:2, :, :]
        inter = ((pred_prob * mask_float) * wb).sum(dim=(2, 3))
        union = ((pred_prob + mask_float) * wb).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        return (wbce + wiou).mean()


class SublabelLoss(nn.Module):
    """
    輔助損失：用於監督原型學習的 K 類交叉熵。
    """
    def __init__(self):
        super(SublabelLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, logits_k_class, sublabel_map):
        # logits_k_class: (B, K, H, W)
        # sublabel_map: (B, H, W)
        
        num_classes = logits_k_class.shape[1]
        target = sublabel_map.clone().long() 
        
        # 1. 處理 0 值 (背景/無效區域)
        ignore_mask = target == 0
        
        # 2. 轉換為 0-indexed (1->0, ..., 8->7)
        target[target > 0] -= 1
        
        # 3. 安全檢查
        invalid_mask = target >= num_classes
        target[ignore_mask | invalid_mask] = -100
        
        # 放大 logits 以增加梯度強度
        return self.ce(logits_k_class * 10.0, target)


class RegularizationLoss(nn.Module):
    """
    無監督正則化損失集合，包含：
    1. Scale Loss (KL Divergence): 防止原型坍塌，鼓勵類別平衡。
    2. Iron Loss (TV Loss): 鼓勵空間平滑。
    3. Knife Loss (Entropy): 鼓勵預測自信度 (可選)。
    """
    def __init__(self, w_scale=0.5, w_iron=0.01, w_knife=0.0):
        """
        Args:
            w_scale (float): Scale Loss 的權重。
            w_iron (float): Iron Loss 的權重。
            w_knife (float): Knife Loss 的權重 (預設為 0，可視需要開啟)。
        """
        super(RegularizationLoss, self).__init__()
        self.w_scale = w_scale
        self.w_iron = w_iron
        self.w_knife = w_knife

    def forward(self, logits_k_class):
        """
        Args:
            logits_k_class (Tensor): (B, K, H, W) 的原始預測分數。
        """
        probs = F.softmax(logits_k_class, dim=1) # (B, K, H, W)
        B, K, H, W = probs.shape
        
        loss_total = 0.0

        # --- 1. Scale Loss (KL Divergence / Balance Loss) ---
        # 目標：讓每個原型在整個 batch 中的平均激發率接近均勻分佈 (1/K)
        if self.w_scale > 0:
            # 計算每個類別在整個 batch 中的平均概率 p_bar: (K,)
            # dim=(0, 2, 3) 對 Batch, Height, Width 求平均
            p_bar = probs.mean(dim=(0, 2, 3)) 
            
            # 目標分佈 t_k = 1/K (均勻分佈)
            t_k = torch.full_like(p_bar, 1.0 / K)
            
            # KL Divergence: sum(p * log(p / t))
            # 添加 1e-8 避免 log(0)
            loss_scale = torch.sum(p_bar * torch.log(p_bar / t_k + 1e-8))
            loss_total += self.w_scale * loss_scale

        # --- 2. Iron Loss (Total Variation / Smoothness Loss) ---
        # 目標：相鄰像素的預測應該相似
        if self.w_iron > 0:
            # 水平方向差異
            h_diff = torch.abs(probs[:, :, :, :-1] - probs[:, :, :, 1:]).mean()
            # 垂直方向差異
            v_diff = torch.abs(probs[:, :, :-1, :] - probs[:, :, 1:, :]).mean()
            
            loss_iron = h_diff + v_diff
            loss_total += self.w_iron * loss_iron

        # --- 3. Knife Loss (Entropy Minimization) ---
        # 目標：讓預測更自信 (更接近 0 或 1)，減少模糊邊界
        if self.w_knife > 0:
            log_probs = F.log_softmax(logits_k_class, dim=1)
            # Entropy = -sum(p * log(p))
            entropy = -torch.sum(probs * log_probs, dim=1) # (B, H, W)
            loss_knife = entropy.mean()
            loss_total += self.w_knife * loss_knife

        return loss_total