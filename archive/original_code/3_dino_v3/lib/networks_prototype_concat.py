import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOv3_Prototype_concat(nn.Module):
    # 將 n_layers 替換為 layer_indices，預設提取第 2, 4, 6, 8, 10, 12 層 (索引 1, 3, 5, 7, 9, 11)
    def __init__(self, num_fg=4, num_bg=4, backbone_type='vitb16', freeze=True, layer_indices=(1, 3, 5, 7, 9, 11)):
        super(DINOv3_Prototype_concat, self).__init__()

        self.freeze = freeze
        self.layer_indices = layer_indices
        
        # 1. 載入 DINOv3 骨幹網路
        print(f"Loading DINOv3: dinov3_{backbone_type}")
        self.backbone = torch.hub.load('facebookresearch/dinov3', f'dinov3_{backbone_type}', source='github', pretrained=False)

        state_dict = torch.load(f"/home/U116med/wch_code/dino_v3/dinov3_{backbone_type}_pretrain.pth", map_location="cpu")
        self.backbone.load_state_dict(state_dict, strict=True)
        
        # 凍結 DINOv3 的所有權重 or not
        for param in self.backbone.parameters():
            param.requires_grad = not self.freeze
            
        # 自動適配單層維度
        if backbone_type in ['vits16', 'vits16plus']:
            self.embed_dim = 384
        elif backbone_type == 'vitb16':
            self.embed_dim = 768
        elif backbone_type == 'vitl16':
            self.embed_dim = 1024
        else:
            raise ValueError(f"不支援的 Backbone: {backbone_type}")
            
        # 計算拼接後的總維度 (單層維度 * 取出的層數)
        self.concat_dim = self.embed_dim * len(self.layer_indices)
            
        self.num_fg = num_fg
        self.num_bg = num_bg
        self.num_total = num_fg + num_bg
        
        # 2. 特徵投影層
        self.decoder_dim = 256 
        self.proj = nn.Sequential(
            nn.Conv2d(self.concat_dim, self.decoder_dim, kernel_size=1),
            nn.BatchNorm2d(self.decoder_dim),
            nn.GELU()
        )
        
        # 3. 可學習的原型 (Prototypes)
        self.prototypes = nn.Parameter(torch.randn(self.num_total, self.decoder_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        B, C, H, W = x.shape
        
        context = torch.no_grad() if self.freeze else torch.enable_grad()
        
        with context:
            # 傳入 layer_indices 列表，精準提取對應深度的特徵
            feats = self.backbone.get_intermediate_layers(x, n=self.layer_indices, reshape=True, norm=True)
            
        # 在通道維度(dim=1)拼接所有特徵: (B, concat_dim, H/16, W/16)
        feat_concat = torch.cat(feats, dim=1)
            
        # 通過投影層將高維特徵壓縮回 decoder_dim
        pixel_embeddings = self.proj(feat_concat) 
        
        # ==========================================
        # Prototype Matching 邏輯
        # ==========================================
        embeddings_norm = F.normalize(pixel_embeddings, p=2, dim=1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)
        
        similarity_map = F.conv2d(embeddings_norm, prototypes_norm.unsqueeze(-1).unsqueeze(-1))
        
        fg_score, _ = torch.max(similarity_map[:, :self.num_fg], dim=1)
        bg_score, _ = torch.max(similarity_map[:, self.num_fg:], dim=1)
        
        logits_small = torch.stack([bg_score, fg_score], dim=1)
        
        logits_high = F.interpolate(logits_small, size=(H, W), mode='bilinear', align_corners=False)
        similarity_map_high = F.interpolate(similarity_map, size=(H, W), mode='bilinear', align_corners=False)
        
        return logits_high, similarity_map_high