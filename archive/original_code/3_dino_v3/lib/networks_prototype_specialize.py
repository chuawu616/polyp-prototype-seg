import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DINOv3_Prototype_Specialize(nn.Module):
    def __init__(self, num_fg=4, num_hard=1, num_bg=4, backbone_type='vits16plus', freeze=True, layer_indices=(1, 3, 5, 7, 9, 11)):
        super(DINOv3_Prototype_Specialize, self).__init__()

        self.freeze = freeze
        self.layer_indices = layer_indices
        
        # 確保前景原型數量至少為 2，才能拆分出 Easy 與 Hard
        assert num_fg >= num_hard + 1, "num_fg 必須大於等於 2，才能拆分為通用與困難原型"
        
        self.num_easy = num_fg - num_hard
        self.num_hard = num_hard
        self.num_bg = num_bg
        self.num_total = num_fg + num_bg
        
        # ==========================================
        # 1. 骨幹網路 (Backbone)
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
        # 3. 解耦的原型定義 (Decoupled Prototypes)
        # ==========================================
        # 將原本的 self.prototypes 拆分為三個獨立的參數矩陣
        self.easy_prototypes = nn.Parameter(torch.randn(self.num_easy, self.decoder_dim))
        self.hard_prototypes = nn.Parameter(torch.randn(self.num_hard, self.decoder_dim))
        self.bg_prototypes = nn.Parameter(torch.randn(self.num_bg, self.decoder_dim))
        
        nn.init.xavier_uniform_(self.easy_prototypes)
        nn.init.xavier_uniform_(self.hard_prototypes)
        nn.init.xavier_uniform_(self.bg_prototypes)
        
        # 溫度係數 (動態放大相似度)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x, alpha=1.0):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        B, C, H, W = x.shape
        
        context = torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            feats = self.backbone.get_intermediate_layers(x, n=self.layer_indices, reshape=True, norm=True)
        feat_concat = torch.cat(feats, dim=1) # (B, concat_dim, H/16, W/16)
        
        pixel_embeddings = self.proto_proj(feat_concat) 
        
        # ==========================================
        # 4. 向量正規化與相似度計算
        # ==========================================
        embeddings_norm = F.normalize(pixel_embeddings, p=2, dim=1)
        easy_norm = F.normalize(self.easy_prototypes, p=2, dim=1)
        hard_norm = F.normalize(self.hard_prototypes, p=2, dim=1)
        bg_norm = F.normalize(self.bg_prototypes, p=2, dim=1)
        
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        
        sim_easy = F.conv2d(embeddings_norm, easy_norm.unsqueeze(-1).unsqueeze(-1)) * logit_scale
        sim_hard = F.conv2d(embeddings_norm, hard_norm.unsqueeze(-1).unsqueeze(-1)) * logit_scale
        sim_bg = F.conv2d(embeddings_norm, bg_norm.unsqueeze(-1).unsqueeze(-1)) * logit_scale
        
        # 使用 max 進行各組內部的特徵聚合
        score_easy, _ = torch.max(sim_easy, dim=1)
        score_bg, _ = torch.max(sim_bg, dim=1)
        score_hard, _ = torch.max(sim_hard, dim=1)
        
        # ==========================================
        # 5. 殘差級聯門控機制 (Residual Cascade Gating)
        # ==========================================
        # 計算通用原型的相對信心：當 score_easy 遠大於 score_bg 時，機率趨近 1
        prob_easy = torch.sigmoid(score_easy - score_bg)
        
        # 困難原型的激活條件：
        # 如果 prob_easy 接近 1 (簡單特徵，判定為息肉)，(1 - prob_easy) 接近 0 -> 壓抑 Hard 原型
        # 如果 prob_easy 接近 0 (模糊特徵，判定為背景)，(1 - prob_easy) 接近 1 -> 釋放 Hard 原型進行誤差修正
        score_fg_final = score_easy + (1.0 - prob_easy) * score_hard * alpha
        
        # 將最終的 BG 與 FG 堆疊
        logits_small = torch.stack([score_bg, score_fg_final], dim=1) # (B, 2, H/16, W/16)
        
        # ==========================================
        # 6. 上採樣與輸出
        # ==========================================
        logits_high = F.interpolate(logits_small, size=(H, W), mode='bilinear', align_corners=False)
        
        # 為了相容之前的視覺化腳本，我們將所有的相似度圖合併回傳
        # 順序為：Easy Prototypes -> Hard Prototype -> BG Prototypes
        sim_all_small = torch.cat([sim_easy, sim_hard, sim_bg], dim=1)
        sim_map_high = F.interpolate(sim_all_small, size=(H, W), mode='bilinear', align_corners=False)
        
        return logits_high, sim_map_high

if __name__ == '__main__':
    model = DINOv3_Prototype_Specialize(num_fg=4, num_bg=4).cuda()
    input_tensor = torch.randn(2, 3, 352, 352).cuda()
    logits_high, sim_map_high = model(input_tensor)
    
    print(f"Logits shape: {logits_high.size()}")
    print(f"Combined Sim_map shape: {sim_map_high.size()}")