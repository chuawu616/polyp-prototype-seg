import torch
import torch.nn as nn
from lib.pvtv2 import pvt_v2_b2

class ProjectionHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.net(x)

class PixelContrastiveModel(nn.Module):
    def __init__(self, pretrained=True, target_stages=[0, 1, 2, 3], proj_dim=128):
        super(LocalContrastiveModel, self).__init__()
        
        # 1. 初始化 Encoder
        self.encoder = pvt_v2_b2(pretrained=pretrained)
        self.target_stages = target_stages
        
        # 2. 為每個需要計算損失的 stage 創建一個獨立的投影頭
        #    獲取每個 stage 的輸出通道數
        encoder_dims = [64, 128, 320, 512]
        
        self.projection_heads = nn.ModuleList()
        for stage_idx in self.target_stages:
            in_dim = encoder_dims[stage_idx]
            # 投影頭的中間層維度可以與輸入相同，輸出維度統一為 proj_dim
            self.projection_heads.append(
                ProjectionHead(in_channels=in_dim, hidden_channels=in_dim, out_channels=proj_dim)
            )

    def forward(self, x):
        """
        前向傳播。
        
        Returns:
            list[Tensor]: 一個列表，包含了每個目標 stage 的特征經過投影頭後得到的【投影特征 z】。
        """
        # a. 通過 Encoder 得到多尺度原始特征 f
        all_features = self.encoder.forward_features(x)
        
        # b. 篩選出我們需要的目標 stage 的原始特征
        target_features = [all_features[i] for i in self.target_stages]
        
        # c. 將每個原始特征 f 通過其對應的投影頭，得到投影特征 z
        output_projections = []
        for i, f in enumerate(target_features):
            z = self.projection_heads[i](f)
            output_projections.append(z)
        
        return output_projections