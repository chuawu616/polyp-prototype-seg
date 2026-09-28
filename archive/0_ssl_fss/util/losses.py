import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryLoss(nn.Module):
    def __init__(self, device='cuda'):
        super(BoundaryLoss, self).__init__()
        # 定义 Sobel 滤波器核
        # 检测水平边缘
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                    dtype=torch.float32, device=device).view(1, 1, 3, 3)
        # 检测垂直边缘
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                    dtype=torch.float32, device=device).view(1, 1, 3, 3)

    def forward(self, pred_probs, true_mask):
        """
        Args:
            pred_probs (Tensor): 模型的概率输出 (B, C, H, W)，通常是 softmax 后的结果。
                                 我们只关心前景类的概率。
            true_mask (Tensor): 真实的二元标签 (B, H, W)，long 类型。
        
        Returns:
            Tensor: 计算出的边界损失。
        """
        # 1. 准备输入
        # 我们只关心前景类 (class 1) 的边界
        foreground_probs = pred_probs[:, 1, :, :].unsqueeze(1) # (B, 1, H, W)
        
        # 将真实标签转换为 float 类型并增加通道维度
        true_mask_float = true_mask.unsqueeze(1).float() # (B, 1, H, W)

        # 2. 使用 Sobel 滤波器计算梯度 (边缘)
        pred_grad_x = F.conv2d(foreground_probs, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(foreground_probs, self.sobel_y, padding=1)
        
        true_grad_x = F.conv2d(true_mask_float, self.sobel_x, padding=1)
        true_grad_y = F.conv2d(true_mask_float, self.sobel_y, padding=1)

        # 3. 计算梯度幅值 (边缘强度)
        pred_edge = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-6)
        true_edge = torch.sqrt(true_grad_x**2 + true_grad_y**2 + 1e-6)

        # 4. 计算 L1 损失
        # 我们希望预测的边缘图和真实的边缘图尽可能接近
        loss = F.l1_loss(pred_edge, true_edge)
        
        return loss