# 檔案: models/PrototypeSegmenter_Pure.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .pvtv2 import pvt_v2_b2
    from .fpn_decoder import FPNDecoder
except ImportError:
    from models.pvtv2 import pvt_v2_b2
    from models.fpn_decoder import FPNDecoder

class PrototypeSegmenterPure(nn.Module):
    def __init__(self, cfg):
        super(PrototypeSegmenterPure, self).__init__()
        
        # 1. Encoder (加載 Stage2 權重)
        self.encoder = pvt_v2_b2(pretrained=False)
        stage2_weight_path = cfg.get('stage2_pretrained_path', None)
        if stage2_weight_path and os.path.exists(stage2_weight_path):
            print(f"###### 正在從 {stage2_weight_path} 加載第二階段 Encoder 權重 ######")
            state_dict = torch.load(stage2_weight_path, map_location='cpu')
            self.encoder.load_state_dict(state_dict, strict=True)
        else:
            print("###### 警告: 使用 ImageNet 預訓練權重 ######")
            self.encoder = pvt_v2_b2(pretrained=True)

        # 2. Decoder
        encoder_out_channels = [64, 128, 320, 512]
        decoder_out_channels = cfg.get('decoder_out_channels', 256)
        self.decoder = FPNDecoder(in_channels=encoder_out_channels, out_channels=decoder_out_channels)
        
        # 3. 原型 (Prototypes)
        self.num_fg = cfg.get('num_fg_prototypes', 4)
        self.num_bg = cfg.get('num_bg_prototypes', 4)
        self.num_total = self.num_fg + self.num_bg
        
        # 隨機初始化原型，讓它們在訓練中學習
        # 形狀: (K, C)
        self.prototypes = nn.Parameter(torch.randn(self.num_total, decoder_out_channels))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, x):
        input_size = x.shape[-2:]
        
        # a. 特徵提取
        features_list = self.encoder.forward_features(x)
        pixel_embeddings = self.decoder(features_list) # (B, C, H', W')
        
        # b. 原型相似度計算
        embeddings_norm = F.normalize(pixel_embeddings, p=2, dim=1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)
        
        # similarity_map: (B, K, H', W')
        # 這就是我們的 K 通道 Logits
        logits_k_class = F.conv2d(embeddings_norm, prototypes_norm.unsqueeze(-1).unsqueeze(-1))
        
        # c. 生成二元 Logits (用於 Structure Loss)
        # 假設前 num_fg 個是前景，後 num_bg 個是背景
        fg_logits = logits_k_class[:, :self.num_fg, :, :]
        bg_logits = logits_k_class[:, self.num_fg:, :, :]
        
        # 取最大值聚合
        fg_score, _ = torch.max(fg_logits, dim=1)
        bg_score, _ = torch.max(bg_logits, dim=1)
        
        logits_binary = torch.stack([bg_score, fg_score], dim=1) # (B, 2, H', W')
        
        # d. 上採樣
        logits_k_class_up = F.interpolate(logits_k_class, size=input_size, mode='bilinear', align_corners=False)
        logits_binary_up = F.interpolate(logits_binary, size=input_size, mode='bilinear', align_corners=False)

        return logits_binary_up, logits_k_class_up