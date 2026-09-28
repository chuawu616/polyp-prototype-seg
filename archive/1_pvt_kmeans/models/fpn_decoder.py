import torch
import torch.nn as nn
import torch.nn.functional as F

class FPNDecoder(nn.Module):
    """
    一個簡單的 FPN 風格的解碼器。
    它接收來自 Encoder 不同 Stage 的特徵列表，並透過一個自頂向下的路徑
    將它們融合成一個高解析度的特徵圖。
    """
    def __init__(self, in_channels, out_channels=256):
        """
        初始化 FPN 解碼器。

        Args:
            in_channels (list[int]): 一個列表，包含了來自 Encoder 各個 Stage 的
                                     輸出通道數。例如，對於 PVTv2-b2，它是
                                     [64, 128, 320, 512]。
            out_channels (int): FPN 內部以及最終輸出的特徵圖通道數。
        """
        super(FPNDecoder, self).__init__()
        self.out_channels = out_channels
        
        # --- 橫向連接 (Lateral Connections) ---
        # 使用 1x1 卷積將每個 Stage 的特徵圖通道數統一到 out_channels。
        # 這一步是為了讓不同層級的特徵能夠相加。
        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels:
            self.lateral_convs.append(
                nn.Conv2d(in_ch, out_channels, kernel_size=1)
            )
            
        # --- 輸出卷積層 ---
        # 在每個融合步驟後，使用一個 3x3 卷積來平滑特徵，消除上採樣可能帶來的混疊效應。
        # 我們只需要 (層數 - 1) 個這樣的卷積層。
        self.output_convs = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.output_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )

    def forward(self, features_list):
        """
        FPN 的前向傳播。

        Args:
            features_list (list[Tensor]): 來自 Encoder 的多尺度特徵圖列表，
                                          順序從淺到深 (解析度從高到低)。
                                          例如 [stage1_feat, stage2_feat, stage3_feat, stage4_feat]。

        Returns:
            Tensor: 融合後的最高解析度特徵圖。
        """
        # 1. 自底向上應用橫向連接
        #    將所有 stage 的特徵都轉換為具有 out_channels 個通道。
        lat_features = [
            self.lateral_convs[i](features_list[i])
            for i in range(len(features_list))
        ]
        
        # 2. 自頂向下進行融合
        #    從最深層的特徵 (p4) 開始
        #    lat_features 的順序是 [p1, p2, p3, p4] (解析度從高到低)
        #    所以我們從列表的末尾開始處理
        fused_feature = lat_features[-1] # 最深層的特徵，例如 p4
        
        # 倒序遍歷橫向特徵和輸出卷積層，進行融合
        # range(len(features_list) - 2, -1, -1) -> 例如對於4層，會產生 2, 1, 0
        for i in range(len(features_list) - 2, -1, -1):
            # a. 將當前的融合特徵上採樣到上一層的尺寸
            upsampled_feature = F.interpolate(fused_feature, 
                                              size=lat_features[i].shape[-2:], # 上一層的 H, W
                                              mode='bilinear', 
                                              align_corners=False)
            
            # b. 將上採樣的特徵與上一層的橫向特徵相加
            fused_feature = lat_features[i] + upsampled_feature
            
            # c. 應用 3x3 卷積進行平滑
            fused_feature = self.output_convs[i](fused_feature)
            
        # 最終的 fused_feature 是融合了所有層級資訊的、解析度最高的特徵圖 (p1)
        return fused_feature