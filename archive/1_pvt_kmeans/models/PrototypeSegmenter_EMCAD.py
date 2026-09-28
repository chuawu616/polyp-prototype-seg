# 檔案: models/PrototypeSegmenter_EMCAD.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .pvtv2 import pvt_v2_b2
    from .emcad_decoder import CASCADE 
except ImportError:
    from models.pvtv2 import pvt_v2_b2
    from models.decoders_emcad import CASCADE

class PrototypeSegmenterEMCAD(nn.Module):
    def __init__(self, cfg):
        super(PrototypeSegmenterEMCAD, self).__init__()
        
        # 1. Encoder (與之前相同)
        self.encoder = pvt_v2_b2(pretrained=False)
        stage2_weight_path = cfg.get('stage2_pretrained_path', None)
        if stage2_weight_path and os.path.exists(stage2_weight_path):
            print(f"###### 正在從 {stage2_weight_path} 加載第二階段 Encoder 權重 ######")
            state_dict = torch.load(stage2_weight_path, map_location='cpu')
            self.encoder.load_state_dict(state_dict, strict=True)
        else:
            print("###### 警告: 使用 ImageNet 預訓練權重 ######")
            self.encoder = pvt_v2_b2(pretrained=True)

        # 2. Decoder (使用 CASCADE)
        # PVTv2-b2 channels: [64, 128, 320, 512] (Stage 1 to 4)
        # CASCADE expects channels from deep to shallow: [512, 320, 128, 64]
        encoder_channels = [64, 128, 320, 512]
        decoder_channels = encoder_channels[::-1] # 反轉列表
        
        self.decoder = CASCADE(channels=decoder_channels)
        
        # CASCADE 輸出的特徵維度等於最淺層的通道數 (64)
        self.decoder_out_channels = decoder_channels[-1] 
        
        # 3. 原型 (Prototypes)
        self.num_fg = cfg.get('num_fg_prototypes', 4)
        self.num_bg = cfg.get('num_bg_prototypes', 4)
        self.num_total = self.num_fg + self.num_bg
        
        # 原型維度必須與 Decoder 輸出維度匹配
        self.prototypes = nn.Parameter(torch.randn(self.num_total, self.decoder_out_channels))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, x):
        input_size = x.shape[-2:]
        
        # a. 提取特徵 [p1, p2, p3, p4]
        features_list = self.encoder.forward_features(x)
        
        # b. 準備 Decoder 輸入
        # x: 最深層 (p4)
        # skips: [p3, p2, p1]
        x_feat = features_list[-1]
        skips = features_list[-2::-1] # 倒序切片，取前三個並反轉
        
        # c. Decoder 前向傳播
        # CASCADE 返回 (dd4, dd3, dd2, dd1, d1)
        # 我們只取最後一個 d1，它是融合了所有信息的最高分辨率特徵
        decoder_outs = self.decoder(x_feat, skips)
        pixel_embeddings = decoder_outs[-1] # (B, 64, H/4, W/4)
        
        # d. 原型相似度計算 (與之前相同)
        embeddings_norm = F.normalize(pixel_embeddings, p=2, dim=1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)
        
        logits_k_class = F.conv2d(embeddings_norm, prototypes_norm.unsqueeze(-1).unsqueeze(-1))
        
        # e. 生成二元 Logits
        fg_logits = logits_k_class[:, :self.num_fg, :, :]
        bg_logits = logits_k_class[:, self.num_fg:, :, :]
        
        fg_score, _ = torch.max(fg_logits, dim=1)
        bg_score, _ = torch.max(bg_logits, dim=1)
        
        logits_binary = torch.stack([bg_score, fg_score], dim=1)
        
        # f. 上採樣
        logits_k_class_up = F.interpolate(logits_k_class, size=input_size, mode='bilinear', align_corners=False)
        logits_binary_up = F.interpolate(logits_binary, size=input_size, mode='bilinear', align_corners=False)

        return logits_binary_up, logits_k_class_up