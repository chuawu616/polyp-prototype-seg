import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOv3_MLP_concat(nn.Module):
    def __init__(self, n_class=1, backbone_type='vitb16', freeze=True, layer_indices=(1, 3, 5, 7, 9, 11)):
        super(DINOv3_MLP_concat, self).__init__()

        self.freeze = freeze
        self.layer_indices = layer_indices
        
        # 1. 載入 DINOv3 骨幹網路
        print(f"Loading DINOv3: dinov3_{backbone_type}")
        self.backbone = torch.hub.load('facebookresearch/dinov3', f'dinov3_{backbone_type}', source='github', pretrained=False)

        state_dict = torch.load(f"/home/U116med/wch_code/dino_v3/dinov3_{backbone_type}_pretrain.pth", map_location="cpu")
        self.backbone.load_state_dict(state_dict, strict=True)
        
        # 2. 凍結 DINOv3 的所有權重 or not
        for param in self.backbone.parameters():
            param.requires_grad = not self.freeze
            
        if backbone_type in ['vits16', 'vits16plus']:
            self.embed_dim = 384
        elif backbone_type == 'vitb16':
            self.embed_dim = 768
        elif backbone_type == 'vitl16':
            self.embed_dim = 1024
            
        # 計算拼接後的總維度
        self.concat_dim = self.embed_dim * len(self.layer_indices)
        
        # 3. 定義簡單的 MLP 分割頭
        self.head = nn.Sequential(
            nn.Linear(self.concat_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_class)
        )

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        B, C, H, W = x.shape
        h_patches = H // 16
        w_patches = W // 16
        
        context = torch.no_grad() if self.freeze else torch.enable_grad()
        
        with context:
            # 傳入層數索引列表
            feats = self.backbone.get_intermediate_layers(x, n=self.layer_indices, reshape=True, norm=True)
            
        # 在通道維度拼接: (B, concat_dim, H/16, W/16)
        feat_concat = torch.cat(feats, dim=1)
            
        # 維度轉換以配合 Linear 層
        feat_concat = feat_concat.permute(0, 2, 3, 1) # -> (B, H/16, W/16, concat_dim)
        feat_flat = feat_concat.reshape(-1, self.concat_dim)   # -> (B * h_patches * w_patches, concat_dim)
        
        # 通過 MLP 進行預測
        logits_flat = self.head(feat_flat)      # -> (B * h_patches * w_patches, n_class)
        
        # 將形狀重塑回 2D 空間特徵圖
        logits_2d = logits_flat.view(B, h_patches, w_patches, -1)
        logits_2d = logits_2d.permute(0, 3, 1, 2) # -> (B, n_class, H/16, W/16)
        
        # 雙線性上採樣回原圖尺寸
        out = F.interpolate(logits_2d, size=(H, W), mode='bilinear', align_corners=False)
        
        return out