# 檔案: models/MetricSegmenter.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 假設 pvt_v2 和 fpn_decoder 在可導入的路徑中
try:
    from .pvtv2 import pvt_v2_b2
    from .fpn_decoder import FPNDecoder
except ImportError:
    from pvt_v2 import pvt_v2_b2
    from fpn_decoder import FPNDecoder

class MetricSegmenter(nn.Module):
    def __init__(self, cfg):
        super(MetricSegmenter, self).__init__()
        
        # 1. 初始化 Encoder 並加載第二階段的權重
        self.encoder = pvt_v2_b2(pretrained=False)
        stage2_weight_path = cfg.get('stage2_pretrained_path', None)
        if stage2_weight_path and os.path.exists(stage2_weight_path):
            print(f"###### 正在從 {stage2_weight_path} 加載第二階段 Encoder 權重 ######")
            state_dict = torch.load(stage2_weight_path, map_location='cpu')
            self.encoder.load_state_dict(state_dict, strict=True)
        else:
            print("###### 警告: 未提供第二階段權重，將使用 ImageNet 預訓練權重。######")
            self.encoder = pvt_v2_b2(pretrained=True)

        # 2. 初始化 FPN Decoder
        encoder_out_channels = [64, 128, 320, 512]
        decoder_out_channels = cfg.get('decoder_out_channels', 256)
        self.decoder = FPNDecoder(in_channels=encoder_out_channels, out_channels=decoder_out_channels)
        
        # 3. 將前景原型定義為可學習參數
        self.num_fg_prototypes = cfg.get('num_fg_prototypes', 4)
        feature_dim = decoder_out_channels
        self.fg_prototypes = nn.Parameter(torch.randn(self.num_fg_prototypes, feature_dim))
        nn.init.xavier_uniform_(self.fg_prototypes) # 使用 Xavier 初始化
        
        # 4. 初始化一個簡單的分割頭
        self.segmentation_head = nn.Conv2d(decoder_out_channels, 1, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]
        
        # a. 提取融合後的特徵 (像素級嵌入)
        features_list = self.encoder.forward_features(x)
        pixel_embeddings = self.decoder(features_list)
        
        # b. 計算與前景原型的相似度 (用於度量損失)
        embeddings_norm = F.normalize(pixel_embeddings, p=2, dim=1)
        prototypes_norm = F.normalize(self.fg_prototypes, p=2, dim=1)
        fg_similarity_map = F.conv2d(embeddings_norm, prototypes_norm.unsqueeze(-1).unsqueeze(-1))
        
        # c. 生成最终的二元分割預測
        # 我們讓分割頭直接預測前景 logit
        fg_logits_small = self.segmentation_head(pixel_embeddings)
        
        # 背景 logit 簡單地設為 0
        bg_logits_small = torch.zeros_like(fg_logits_small)
        final_logits_small = torch.cat([bg_logits_small, fg_logits_small], dim=1)

        # d. 上採樣回原始尺寸
        logits_high_res = F.interpolate(final_logits_small, size=input_size, mode='bilinear', align_corners=False)
        
        # 返回最终的二元 logits 和用於計算度量損失的中间结果
        return logits_high_res, fg_similarity_map