import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.pvtv2 import pvt_v2_b2
from lib.decoders import CASCADE

class Prototype_CASCADE_Specialize(nn.Module):
    def __init__(self, num_fg=4, num_hard=2, num_bg=4, encoder_path=None):
        super(Prototype_CASCADE_Specialize, self).__init__()
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
        
        self.easy_prototypes = nn.Parameter(torch.randn(self.num_easy, self.decoder_dim))
        self.hard_prototypes = nn.Parameter(torch.randn(self.num_hard, self.decoder_dim))
        self.bg_prototypes = nn.Parameter(torch.randn(self.num_bg, self.decoder_dim))
        
        nn.init.xavier_uniform_(self.easy_prototypes)
        nn.init.xavier_uniform_(self.hard_prototypes)
        nn.init.xavier_uniform_(self.bg_prototypes)
        
        # 溫度係數 (動態放大相似度)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x, alpha=1.0):
        # Encoder
        x1, x2, x3, x4 = self.backbone(x)
        
        # Decoder
        outs = self.decoder(x4, [x3, x2, x1])
        pixel_embeddings = outs[-1] # (B, 64, H/4, W/4)
        
        # ==========================================
        # 向量正規化與相似度計算
        # ==========================================
        embeddings_norm = F.normalize(pixel_embeddings, p=2, dim=1)
        easy_norm = F.normalize(self.easy_prototypes, p=2, dim=1)
        hard_norm = F.normalize(self.hard_prototypes, p=2, dim=1)
        bg_norm = F.normalize(self.bg_prototypes, p=2, dim=1)
        
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        
        sim_easy = F.conv2d(embeddings_norm, easy_norm.unsqueeze(-1).unsqueeze(-1)) * logit_scale
        sim_hard = F.conv2d(embeddings_norm, hard_norm.unsqueeze(-1).unsqueeze(-1)) * logit_scale
        sim_bg = F.conv2d(embeddings_norm, bg_norm.unsqueeze(-1).unsqueeze(-1)) * logit_scale
        
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
        # 維持總通道數為 K = num_fg + num_bg
        sim_all_small = torch.cat([sim_easy, sim_hard, sim_bg], dim=1)
        
        # 上採樣回原圖尺寸
        logits_high = F.interpolate(logits_small, size=x.shape[-2:], mode='bilinear', align_corners=False)
        similarity_map_high = F.interpolate(sim_all_small, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        return logits_high, similarity_map_high
        
if __name__ == '__main__':
    model = Prototype_CASCADE_Specialize(num_fg=4, num_bg=4).cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    # 模擬訓練初期的呼叫方式 (alpha 較小)
    logits_high, similarity_map_high = model(input_tensor, alpha=0.1)
    
    print(f"Logits shape: {logits_high.size()}")
    print(f"Similarity Map shape: {similarity_map_high.size()}")