import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super(ProjectionHead, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.head(x)

class PixelContrastiveModel(nn.Module):
    def __init__(self, pretrained=True, proj_dim=128):
        super(PixelContrastiveModel, self).__init__()
        
        # 1. Encoder (PVTv2-b2)
        self.encoder = pvt_v2_b2(pretrained=pretrained)
        
        # 2. Projection Heads
        # PVTv2-b2 輸出通道: [64, 128, 320, 512]
        self.proj_heads = nn.ModuleList([
            ProjectionHead(64, proj_dim),
            ProjectionHead(128, proj_dim),
            ProjectionHead(320, proj_dim),
            ProjectionHead(512, proj_dim)
        ])

    def forward(self, x):
        # 提取特徵 [p1, p2, p3, p4]
        features = self.encoder.forward_features(x)
        
        projections = []
        for i, feat in enumerate(features):
            proj = self.proj_heads[i](feat)
            proj = F.normalize(proj, p=2, dim=1)
            projections.append(proj)
            
        return projections