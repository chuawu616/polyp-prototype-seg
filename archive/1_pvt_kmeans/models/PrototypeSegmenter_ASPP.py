# 檔案: models/PrototypeSegmenter_ASPP.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

try:
    from .pvtv2 import pvt_v2_b2
    from .decoders_aspp import DeepLabV3PlusDecoder
except ImportError:
    from models.pvtv2 import pvt_v2_b2
    from models.aspp_decoder import DeepLabV3PlusDecoder

class PrototypeSegmenterASPP(nn.Module):
    def __init__(self, cfg):
        super(PrototypeSegmenterASPP, self).__init__()
        
        # 1. Encoder
        self.encoder = pvt_v2_b2(pretrained=False)
        stage2_weight_path = cfg.get('stage2_pretrained_path', None)
        if stage2_weight_path and os.path.exists(stage2_weight_path):
            print(f"###### 正在從 {stage2_weight_path} 加載第二階段 Encoder 權重 ######")
            state_dict = torch.load(stage2_weight_path, map_location='cpu')
            self.encoder.load_state_dict(state_dict, strict=True)
        else:
            print("###### 警告: 使用 ImageNet 預訓練權重 ######")
            self.encoder = pvt_v2_b2(pretrained=True)

        # 2. Decoder (使用 ASPP)
        encoder_out_channels = [64, 128, 320, 512]
        decoder_out_channels = cfg.get('decoder_out_channels', 256)
        
        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=encoder_out_channels,
            decoder_channels=decoder_out_channels,
            atrous_rates=[6, 12, 18] # 可以根據特徵圖大小調整，對於 11x11 (352/32)，這可能有點大，但對於 Context 捕獲仍有效
        )
        
        # 3. 原型 (Prototypes)
        self.num_fg = cfg.get('num_fg_prototypes', 4)
        self.num_bg = cfg.get('num_bg_prototypes', 4)
        self.num_total = self.num_fg + self.num_bg
        
        self.prototypes = nn.Parameter(torch.randn(self.num_total, decoder_out_channels))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, x):
        input_size = x.shape[-2:]
        
        # a. 提取特徵
        features_list = self.encoder.forward_features(x)
        
        # b. Decoder 融合
        pixel_embeddings = self.decoder(features_list) # (B, 256, H/4, W/4)
        
        # c. 原型相似度
        embeddings_norm = F.normalize(pixel_embeddings, p=2, dim=1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)
        logits_k_class = F.conv2d(embeddings_norm, prototypes_norm.unsqueeze(-1).unsqueeze(-1))
        
        # d. 二元 Logits
        fg_logits = logits_k_class[:, :self.num_fg, :, :]
        bg_logits = logits_k_class[:, self.num_fg:, :, :]
        fg_score, _ = torch.max(fg_logits, dim=1)
        bg_score, _ = torch.max(bg_logits, dim=1)
        logits_binary = torch.stack([bg_score, fg_score], dim=1)
        
        # e. 上採樣
        logits_k_class_up = F.interpolate(logits_k_class, size=input_size, mode='bilinear', align_corners=False)
        logits_binary_up = F.interpolate(logits_binary, size=input_size, mode='bilinear', align_corners=False)

        return logits_binary_up, logits_k_class_up