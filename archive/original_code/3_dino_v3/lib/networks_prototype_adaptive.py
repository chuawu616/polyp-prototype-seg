import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DINOv3_Prototype_Adaptive(nn.Module):
    def __init__(self, pool_size=64, backbone_type='vits16plus', freeze=True, layer_indices=(1, 3, 5, 7, 9, 11)):
        super(DINOv3_Prototype_Adaptive, self).__init__()

        self.freeze = freeze
        self.layer_indices = layer_indices
        
        # 建立龐大的原型池 (不再硬性區分 FG/BG，由網路自適應學習)
        self.pool_size = pool_size 
        
        # ==========================================
        # 1. 骨幹網路
        # ==========================================
        print(f"Loading DINOv3: dinov3_{backbone_type}")
        self.backbone = torch.hub.load('facebookresearch/dinov3', f'dinov3_{backbone_type}', source='github', pretrained=False)
        state_dict = torch.load(f"/home/U116med/wch_code/dino_v3/dinov3_{backbone_type}_pretrain.pth", map_location="cpu")
        self.backbone.load_state_dict(state_dict, strict=True)
        
        for param in self.backbone.parameters():
            param.requires_grad = not self.freeze
            
        if backbone_type in ['vits16', 'vits16plus']:
            self.embed_dim = 384
        elif backbone_type == 'vitb16':
            self.embed_dim = 768
        elif backbone_type == 'vitl16':
            self.embed_dim = 1024
            
        self.concat_dim = self.embed_dim * len(self.layer_indices)
        
        # ==========================================
        # 2. 特徵投影層
        # ==========================================
        self.decoder_dim = 256 
        self.proto_proj = nn.Sequential(
            nn.Conv2d(self.concat_dim, self.decoder_dim, kernel_size=1),
            nn.BatchNorm2d(self.decoder_dim),
            nn.GELU()
        )
        
        # ==========================================
        # 3. [Direction 2] 正交初始化的龐大原型池
        # ==========================================
        self.prototypes = nn.Parameter(torch.empty(self.pool_size, self.decoder_dim))
        # 關鍵：強制所有 Prototype 初始狀態互相正交，確保特徵捕捉的多樣性最大化
        nn.init.orthogonal_(self.prototypes)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # ==========================================
        # 4. [Direction 1] 影像級自適應原型門控 (Image-Conditioned Gate)
        # ==========================================
        # 接收整張影像的全局特徵，決定要開啟哪些 Prototype
        self.adaptive_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.concat_dim, self.pool_size // 2),
            nn.GELU(),
            nn.Linear(self.pool_size // 2, self.pool_size),
            nn.Sigmoid() # 輸出 0~1 之間的激活權重
        )
        
        # 5. 最終的動態聚合層 (將激活的 Prototype 壓縮為最終預測)
        # 由於我們不再寫死前 N 個是 FG，後 M 個是 BG，我們讓 1x1 Conv 學習如何將這些 Prototype 的響應轉為息肉機率
        self.aggregation_conv = nn.Conv2d(self.pool_size, 1, kernel_size=1)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        B, C, H, W = x.shape
        
        context = torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            feats = self.backbone.get_intermediate_layers(x, n=self.layer_indices, reshape=True, norm=True)
        feat_concat = torch.cat(feats, dim=1)
        
        # ------------------------------------------
        # 產生空間特徵與計算 Gate
        # ------------------------------------------
        pixel_embeddings = self.proto_proj(feat_concat) 
        
        # 動態決定這張圖片需要用到哪些 Prototype (B, pool_size)
        gate_weights = self.adaptive_gate(feat_concat) 
        
        # ------------------------------------------
        # 相似度計算與門控遮罩 (Gating Mask)
        # ------------------------------------------
        embeddings_norm = F.normalize(pixel_embeddings, p=2, dim=1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)
        
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        
        # 基礎相似度 (B, pool_size, H/16, W/16)
        raw_sim_map = F.conv2d(embeddings_norm, prototypes_norm.unsqueeze(-1).unsqueeze(-1)) * logit_scale
        
        # 關鍵：將不被需要的 Prototype 相似度抹除
        # 將 gate_weights reshape 為 (B, pool_size, 1, 1) 進行廣播相乘
        gate_mask = gate_weights.unsqueeze(-1).unsqueeze(-1)
        gated_sim_map = raw_sim_map * gate_mask
        
        # ------------------------------------------
        # 聚合為最終預測
        # ------------------------------------------
        # 將篩選過後的 Prototype 響應圖，透過 1x1 Conv 融合成單通道的息肉機率
        logits_small = self.aggregation_conv(gated_sim_map)
        
        # 上採樣
        logits_high = F.interpolate(logits_small, size=(H, W), mode='bilinear', align_corners=False)
        gated_sim_map_high = F.interpolate(gated_sim_map, size=(H, W), mode='bilinear', align_corners=False)
        
        return logits_high, gated_sim_map_high, gate_weights

if __name__ == '__main__':
    # 測試 pool_size=64
    model = DINOv3_Adaptive_Prototypes(pool_size=64, layer_indices=(3, 7, 11)).cuda()
    input_tensor = torch.randn(2, 3, 352, 352).cuda()
    logits, sim_map, gates = model(input_tensor)
    
    print(f"Prediction shape: {logits.size()}")
    print(f"Gated Similarity Map shape: {sim_map.size()}")
    print(f"Gate Weights shape: {gates.size()}")