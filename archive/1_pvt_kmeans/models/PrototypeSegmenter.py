import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .pvtv2 import pvt_v2_b2
from .fpn_decoder import FPNDecoder


class PrototypeSegmenter(nn.Module):
    """
    整合了 PVTv2 Encoder, FPN Decoder 和可學習原型的最終分割模型。
    """
    def __init__(self, cfg):
        super(PrototypeSegmenter, self).__init__()
        
        # 1. 初始化可訓練的 PVTv2 Encoder
        self.encoder = pvt_v2_b2(pretrained=False)

        # --- 新增的本地權重加載邏輯 ---
        # 從設定檔中獲取本地 Encoder 權重的路徑
        encoder_weight_path = cfg.get('encoder_weight_path', None)

        if encoder_weight_path and os.path.exists(encoder_weight_path):
            print(f"###### 正在從本地路徑加載 Encoder 預訓練權重: {encoder_weight_path} ######")
            try:
                # 加載權重檔案
                # map_location='cpu' 是一個好習慣，可以避免 GPU 記憶體問題，加載後再移到指定設備
                state_dict = torch.load(encoder_weight_path, map_location='cpu', weights_only=True)
                
                cleaned_state_dict = {}
                for k, v in state_dict.items():
                    # 移除 'module.'、'backbone.' 或 'encoder.' 等常見前綴
                    new_key = k.replace('module.', '').replace('backbone.', '').replace('encoder.', '')
                    cleaned_state_dict[new_key] = v
                
                # 加載清洗後的權重
                # strict=False 允許部分加載，如果權重檔案中包含模型沒有的鍵 (例如分類頭)，
                # 或者模型有權重檔案中沒有的鍵，程式不會報錯。
                self.encoder.load_state_dict(cleaned_state_dict, strict=False)
                print("###### 本地 Encoder 權重加載成功! ######")

            except Exception as e:
                print(f"!!!!!! 加載本地 Encoder 權重失敗: {e} !!!!!!")
                print("!!!!!! Encoder 將使用隨機初始化權重 !!!!!!")
        else:
            print("###### 未提供有效的本地 Encoder 權重路徑，正在嘗試加載在線 ImageNet 權重... ######")
            # 如果沒有提供本地路徑，則退回到加載在線 ImageNet 權重 (如果需要)
            # 這需要 pvt_v2_b2 內部支持 pretrained=True
            # 為了簡單起見，我們可以在這裡重新加載
            self.encoder = pvt_v2_b2(pretrained=True)
        # 2. 初始化 FPN Decoder
        encoder_out_channels = [64, 128, 320, 512]
        decoder_out_channels = cfg.get('decoder_out_channels', 128)
        self.decoder = FPNDecoder(in_channels=encoder_out_channels, out_channels=decoder_out_channels)
    
        # 3. 載入原型對應的超類標籤 (前景/背景)
        if 'prototype_labels' not in cfg:
            raise ValueError("模型配置 'model_cfg' 中必须提供 'prototype_labels'。")
        prototype_labels = torch.tensor(cfg['prototype_labels']).long()
        self.register_buffer("prototype_labels", prototype_labels)
        
    def forward(self, x, per_image_prototypes):
        # 獲取輸入圖像的原始空間尺寸，例如 (352, 352)
        input_size = x.shape[-2:]
        
        # 1. 提取並融合多尺度特徵
        features_list = self.encoder.forward_features(x)
        fused_features = self.decoder(features_list) # (B, C, H', W'), e.g., (B, 128, 88, 88)
        
        # 2. 原型匹配 (計算余弦相似度)
        features_norm = F.normalize(fused_features, p=2, dim=1)
        prototypes_norm = F.normalize(per_image_prototypes, p=2, dim=2)
        B, C, H, W = features_norm.shape
        N_proto = prototypes_norm.shape[1]
        
        # 將 features_norm reshape 為 (1, B*C, H, W)
        features_reshaped = features_norm.view(1, B * C, H, W)
        
        # 將 prototypes_norm reshape 為 conv2d 的權重格式
        # (B*16, C, 1, 1)
        prototypes_reshaped = prototypes_norm.view(B * N_proto, C, 1, 1)
        
        # 使用 groups=B 進行分組卷積
        # 每個 group (對應 batch 中的一個樣本) 只會看到自己的原型
        similarity_map_flat = F.conv2d(features_reshaped, prototypes_reshaped, groups=B)
        
        # 將結果 reshape 回批次格式
        similarity_map = similarity_map_flat.view(B, N_proto, H, W) # (B, 16, H', W')

        # 3. 生成二元 Logits
        fg_indices = (self.prototype_labels == 1).nonzero(as_tuple=True)[0]
        bg_indices = (self.prototype_labels == 0).nonzero(as_tuple=True)[0]
        
        fg_scores, _ = torch.max(similarity_map[:, fg_indices], dim=1)
        bg_scores, _ = torch.max(similarity_map[:, bg_indices], dim=1)
        
        logits_small = torch.stack([bg_scores, fg_scores], dim=1)
        logits_high_res = F.interpolate(logits_small, size=input_size, mode='bilinear', align_corners=False)
        
        return logits_high_res
