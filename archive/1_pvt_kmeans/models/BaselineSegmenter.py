# 檔案: models/BaselineSegmenter.py

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .pvtv2 import pvt_v2_b2
    from .decoders_emcad import CASCADE
except ImportError:
    from models.pvtv2 import pvt_v2_b2
    from models.decoders_emcad import CASCADE

class BaselineSegmenter(nn.Module):
    """
    一個純粹的 Encoder-Decoder 分割模型。
    Encoder: PVTv2-b2
    Decoder: EMCAD (CASCADE)
    Head: 簡單的 1x1 卷積，輸出 1 個通道 (前景 Logit)
    """
    def __init__(self, pretrained=True):
        super(BaselineSegmenter, self).__init__()
        
        # 1. Encoder
        # 我們直接加載 ImageNet 預訓練權重，這是最強的起點
        self.encoder = pvt_v2_b2(pretrained=pretrained)

        # 2. Decoder (CASCADE)
        # PVTv2-b2 channels: [64, 128, 320, 512]
        # CASCADE expects: [512, 320, 128, 64]
        encoder_channels = [64, 128, 320, 512]
        decoder_channels = encoder_channels[::-1]
        
        self.decoder = CASCADE(channels=decoder_channels)
        
        # 3. Segmentation Head
        # CASCADE 的最終輸出 d1 的通道數等於 encoder 最淺層的通道數 (64)
        self.out_channels = decoder_channels[-1] 
        self.head = nn.Conv2d(self.out_channels, 1, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]
        
        # a. Encoder
        features_list = self.encoder.forward_features(x)
        
        # b. Decoder 準備
        x_feat = features_list[-1]     # p4 (深)
        skips = features_list[-2::-1]  # [p3, p2, p1] (淺)
        
        # c. Decoder
        # CASCADE 返回 tuple，最後一個是 d1
        decoder_outs = self.decoder(x_feat, skips)
        d1 = decoder_outs[-1] # (B, 64, H/4, W/4)
        
        # d. Head
        logits = self.head(d1) # (B, 1, H/4, W/4)
        
        # e. Upsample
        logits_up = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        
        return logits_up # 返回 (B, 1, H, W)