import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOv3_Prototype(nn.Module):
    def __init__(self, num_fg=4, num_bg=4, backbone_type='vitb16', freeze=True):
        super(DINOv3_Prototype, self).__init__()

        self.freeze = freeze
        
        # 1. 載入 DINOv3 骨幹網路
        print(f"Loading DINOv3: dinov3_{backbone_type}")
        self.backbone = torch.hub.load('facebookresearch/dinov3', f'dinov3_{backbone_type}', source='github', pretrained=False)

        state_dict = torch.load(f"/home/U116med/wch_code/dino_v3/dinov3_{backbone_type}_pretrain.pth", map_location="cpu")
        self.backbone.load_state_dict(state_dict, strict=True)
        
        # 凍結 DINOv3 的所有權重 or not
        for param in self.backbone.parameters():
            param.requires_grad = not self.freeze
            
        # 自動適配維度
        if backbone_type in ['vits16', 'vits16plus']:
            self.embed_dim = 384
        elif backbone_type == 'vitb16':
            self.embed_dim = 768
        elif backbone_type == 'vitl16':
            self.embed_dim = 1024
        else:
            raise ValueError(f"不支援的 Backbone: {backbone_type}")
            
        self.num_fg = num_fg
        self.num_bg = num_bg
        self.num_total = num_fg + num_bg
        
        # # 2. 特徵投影層 (將 DINOv3 的高維特徵壓縮，以利原型計算)
        self.decoder_dim = 256 
        self.proj = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.decoder_dim, kernel_size=1),
            nn.BatchNorm2d(self.decoder_dim),
            nn.GELU()
        )
        
        # 3. 可學習的原型 (Prototypes)
        self.prototypes = nn.Parameter(torch.randn(self.num_total, self.decoder_dim))
        
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, x):
        # 如果輸入是單通道(灰階)，將其複製為 3 通道
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        B, C, H, W = x.shape
        
        # 提取 DINOv3 特徵 
        context = torch.no_grad() if self.freeze else torch.enable_grad()
        
        with context:
            feats = self.backbone.get_intermediate_layers(x, n=1, reshape=True, norm=True)
            feat_map = feats[0]
            
        # 通過投影層
        pixel_embeddings = self.proj(feat_map) # (B, decoder_dim, H/16, W/16)
        
        # ==========================================
        # Prototype Matching 邏輯
        # ==========================================
        # 1. 空間特徵與原型都進行 L2 正規化 (投影到單位球面上)
        embeddings_norm = F.normalize(pixel_embeddings, p=2, dim=1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)
        
        # 2. 計算餘弦相似度 (Cosine Similarity)
        # 這裡巧妙利用 1x1 卷積來計算每個像素與所有 K 個原型的內積
        # 輸出形狀: (B, num_total, H/16, W/16)
        similarity_map = F.conv2d(embeddings_norm, prototypes_norm.unsqueeze(-1).unsqueeze(-1))
        
        # 3. 生成二元 Logits (取各個子類別中的最大相似度代表該像素屬於前景或背景的機率)
        fg_score, _ = torch.max(similarity_map[:, :self.num_fg], dim=1)
        bg_score, _ = torch.max(similarity_map[:, self.num_fg:], dim=1)
        # fg_score = torch.logsumexp(similarity_map[:, :self.num_fg], dim=1)
        # bg_score = torch.logsumexp(similarity_map[:, self.num_fg:], dim=1)
        
        # 將 BG 與 FG 堆疊起來: (B, 2, H/16, W/16)
        # Channel 0: BG, Channel 1: FG
        logits_small = torch.stack([bg_score, fg_score], dim=1)
        
        # 4. 雙線性上採樣回原始輸入影像的大小 (H, W)
        logits_high = F.interpolate(logits_small, size=(H, W), mode='bilinear', align_corners=False)
        similarity_map_high = F.interpolate(similarity_map, size=(H, W), mode='bilinear', align_corners=False)
        
        # 返回:
        # 1. 二元 Logits (用於 Structure Loss 計算與預測)
        # 2. K類相似度圖 (用於後續的視覺化與正則化)
        return logits_high, similarity_map_high

if __name__ == '__main__':
    model = DINOv3_Prototype().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()
    logits_high, similarity_map_high = model(input_tensor)
    print(f"Logits shape: {logits_high.size()}, Sim_map shape: {similarity_map_high.size()}")