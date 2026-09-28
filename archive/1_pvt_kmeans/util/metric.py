import numpy as np
import torch

class SegmentationMetric:
    """
    計算二元或多類別分割任務的指標，如 Dice 和 IoU。
    """
    def __init__(self, num_classes):
        """
        初始化指標計算器。

        Args:
            num_classes (int): 類別總數 (包含背景)。對於二元分割，通常為 2。
        """
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """
        重置所有累計的統計數據。
        """
        # 創建一個混淆矩陣 (confusion matrix) 來累計結果
        # self.hist 的 shape 是 (num_classes, num_classes)
        # self.hist[i, j] 代表真實標籤為 i，但被預測為 j 的像素總數
        self.hist = np.zeros((self.num_classes, self.num_classes))

    def _fast_hist(self, label_true, label_pred):
        """
        快速計算單張圖像的混淆矩陣。
        """
        # 排除掉被忽略的標籤 (例如邊界像素)
        mask = (label_true >= 0) & (label_true < self.num_classes)
        # 展平陣列並計算混淆矩陣
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def update(self, label_trues, label_preds):
        """
        使用一批新的預測和真實標籤來更新混淆矩陣。

        Args:
            label_trues (Tensor or np.array): 真實標籤，形狀 (B, H, W)。
            label_preds (Tensor or np.array): 預測標籤，形狀 (B, H, W)。
        """
        # 如果是 PyTorch Tensor，先轉換為 NumPy 陣列
        if isinstance(label_trues, torch.Tensor):
            label_trues = label_trues.cpu().numpy()
        if isinstance(label_preds, torch.Tensor):
            label_preds = label_preds.cpu().numpy()

        for lt, lp in zip(label_trues, label_preds):
            self.hist += self._fast_hist(lt.flatten(), lp.flatten())

    def get_scores(self):
        """
        根據累計的混淆矩陣計算各種指標。

        Returns:
            dict: 包含各種指標的字典。
        """
        # 從混淆矩陣中獲取 TP, FP, FN
        # True Positives (TP) 是對角線上的元素
        tp = np.diag(self.hist)
        # False Positives (FP) 是每列的總和減去 TP
        fp = self.hist.sum(axis=0) - tp
        # False Negatives (FN) 是每行的總和減去 TP
        fn = self.hist.sum(axis=1) - tp
        
        # --- 計算 IoU (Intersection over Union) ---
        # IoU = TP / (TP + FP + FN)
        iou = tp / (tp + fp + fn + 1e-8) # 添加平滑項防止除以零
        # 平均 IoU (mIoU)
        mean_iou = np.nanmean(iou)

        # --- 計算 Dice Coefficient ---
        # Dice = 2 * TP / (2 * TP + FP + FN)
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        # 平均 Dice
        mean_dice = np.nanmean(dice)

        # --- 計算像素準確率 (Pixel Accuracy) ---
        acc = tp.sum() / (self.hist.sum() + 1e-8)
        
        # 返回一個包含所有指標的字典
        return {
            "Pixel_Accuracy": acc,
            "Mean_IoU": mean_iou,
            "Mean_Dice": mean_dice,
            "Class_IoU": iou,
            "Class_Dice": dice,
        }