import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.pvtv2 import pvt_v2_b2
from lib.decoders import CASCADE

class DynamicHardPrototypes(nn.Module):
    def __init__(self, num_hard, in_channels, decoder_dim):
        super(DynamicHardPrototypes, self).__init__()
        self.num_hard = num_hard
        self.decoder_dim = decoder_dim
        
        # 1. 全局錨點 (Global Anchor) - 提供訓練初期的穩定性
        self.global_hard = nn.Parameter(torch.randn(num_hard, decoder_dim))
        nn.init.xavier_uniform_(self.global_hard)
        
        # 2. 動態特徵提取器 - 從當前特徵圖預測聚類權重
        self.dynamic_extractor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_hard, kernel_size=1)
        )
        
        # 3. 殘差門控 (初始化為 0) - 控制動態特徵的介入程度
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, features, alpha=1.0):
        """
        features: (B, C, H, W) 這裡接收的是 CASCADE 輸出的 64 維特徵
        """
        B, C, H, W = features.shape
        
        # --- 動態 Prototype 生成 ---
        # 預測每個像素屬於哪個 hard prototype 的權重
        weights = self.dynamic_extractor(features) # (B, num_hard, H, W)
        weights = weights.view(B, self.num_hard, -1) # (B, num_hard, H*W)
        
        # 空間維度 Softmax，使每個 Prototype 的權重總和為 1
        attn_weights = F.softmax(weights, dim=-1) 
        
        # 萃取動態 Prototype (將特徵圖依照權重加權平均)
        features_flat = features.view(B, C, -1).transpose(1, 2) # (B, H*W, C)
        dynamic_proto = torch.bmm(attn_weights, features_flat)  # (B, num_hard, C)
        
        # --- 全局與動態融合 ---
        # 將全局錨點擴充 Batch 維度
        global_proto = self.global_hard.unsqueeze(0).expand(B, -1, -1) 
        
        # 透過 gamma 進行殘差融合
        final_hard_proto = global_proto + self.gamma * dynamic_proto * alpha
        
        return final_hard_proto


class Prototype_CASCADE_Hybrid_Specialize(nn.Module):
    def __init__(self, num_fg=4, num_hard=2, num_bg=4, encoder_path=None):
        super(Prototype_CASCADE_Hybrid_Specialize, self).__init__()
        self.backbone = pvt_v2_b2()
        path = '/home/U116med/wch_code/cascade/weights/pvt_v2_b2.pth'
        if encoder_path:
            path = encoder_path
        
        # 載入預訓練權重
        save_model = torch.load(path, weights_only=True)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
            
        self.decoder = CASCADE(channels=[512, 320, 128, 64])
        print('Model %s created, param count: %d' %
                     ('decoder: ', sum([m.numel() for m in self.decoder.parameters()])))
        
        # ==========================================
        # 解耦的原型定義 (Decoupled Prototypes)
        # ==========================================
        assert num_fg >= num_hard + 1, "num_fg 必須大於等於 2，才能拆分為通用與困難原型"
        
        self.num_easy = num_fg - num_hard
        self.num_hard = num_hard
        self.num_bg = num_bg
        self.num_total = num_fg + num_bg
        
        self.decoder_dim = 64 
        
        # Easy 與 BG 維持全局靜態
        self.easy_prototypes = nn.Parameter(torch.randn(self.num_easy, self.decoder_dim))
        self.bg_prototypes = nn.Parameter(torch.randn(self.num_bg, self.decoder_dim))
        nn.init.xavier_uniform_(self.easy_prototypes)
        nn.init.xavier_uniform_(self.bg_prototypes)
        
        # Hard 替換為 Hybrid 模組
        self.hard_prototypes_module = DynamicHardPrototypes(
            num_hard=self.num_hard, 
            in_channels=self.decoder_dim, 
            decoder_dim=self.decoder_dim
        )
        
        # 溫度係數 (動態放大相似度)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x, alpha=1.0):
        # Encoder
        x1, x2, x3, x4 = self.backbone(x)
        
        # Decoder
        outs = self.decoder(x4, [x3, x2, x1])
        pixel_embeddings = outs[-1] # (B, 64, H/4, W/4)
        
        # ==========================================
        # 獲取 Prototypes 並正規化
        # ==========================================
        # 1. 靜態 Prototypes (Easy, BG)
        easy_norm = F.normalize(self.easy_prototypes, p=2, dim=1)
        bg_norm = F.normalize(self.bg_prototypes, p=2, dim=1)
        
        # 2. 動態混合 Prototypes (Hard) - 帶有 Batch 維度 (B, num_hard, 64)
        hard_hybrid = self.hard_prototypes_module(pixel_embeddings, alpha)
        hard_norm = F.normalize(hard_hybrid, p=2, dim=-1)
        
        # 特徵圖正規化
        embeddings_norm = F.normalize(pixel_embeddings, p=2, dim=1)
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        
        # ==========================================
        # 相似度計算
        # ==========================================
        # Easy 與 BG 沒有 Batch 維度，用 conv2d 即可
        sim_easy = F.conv2d(embeddings_norm, easy_norm.unsqueeze(-1).unsqueeze(-1)) * logit_scale
        sim_bg = F.conv2d(embeddings_norm, bg_norm.unsqueeze(-1).unsqueeze(-1)) * logit_scale
        
        # Hard 帶有 Batch 維度，使用 einsum 進行矩陣乘法
        # embeddings_norm: (B, C, H, W)
        # hard_norm:       (B, N, C) 
        # Output sim_hard: (B, N, H, W)
        sim_hard = torch.einsum('bchw,bnc->bnhw', embeddings_norm, hard_norm) * logit_scale
        
        # ==========================================
        # 特徵聚合與門控機制
        # ==========================================
        score_easy, _ = torch.max(sim_easy, dim=1)
        score_bg, _ = torch.max(sim_bg, dim=1)
        score_hard, _ = torch.max(sim_hard, dim=1) 
        
        prob_easy = torch.sigmoid(score_easy - score_bg)
        
        # 結合 alpha 退火係數的殘差修正
        score_fg_final = score_easy + (1.0 - prob_easy) * alpha * score_hard
        
        # ==========================================
        # 輸出處理
        # ==========================================
        # Stack to (B, 2, H/4, W/4) -> Channel 0: BG, Channel 1: FG
        logits_small = torch.stack([score_bg, score_fg_final], dim=1)
        
        # 為了視覺化腳本相容，依序合併 [Easy, Hard, BG] 
        sim_all_small = torch.cat([sim_easy, sim_hard, sim_bg], dim=1)
        
        # 上採樣回原圖尺寸
        logits_high = F.interpolate(logits_small, size=x.shape[-2:], mode='bilinear', align_corners=False)
        similarity_map_high = F.interpolate(sim_all_small, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        return logits_high, similarity_map_high
        
if __name__ == '__main__':
    # 測試程式碼
    model = Prototype_CASCADE_Hybrid_Specialize(num_fg=4, num_hard=2, num_bg=4).cuda()
    input_tensor = torch.randn(2, 3, 352, 352).cuda() # 測試 Batch size = 2

    # 模擬訓練初期的呼叫方式 (alpha 較小)
    logits_high, similarity_map_high = model(input_tensor, alpha=0.1)
    
    print(f"Logits shape: {logits_high.size()}")
    print(f"Similarity Map shape: {similarity_map_high.size()}")