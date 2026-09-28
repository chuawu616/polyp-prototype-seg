import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2
from lib.decoders import CASCADE

class Prototype_CASCADE(nn.Module):
    def __init__(self, num_fg=4, num_bg=4, encoder_path=None):
        super(Prototype_CASCADE, self).__init__()
        self.backbone = pvt_v2_b2()
        path = '/home/U116med/wch_code/cascade/weights/pvt_v2_b2.pth'
        #path = '/home/U116med/wch_code/cascade/models/pretrain_v1/pretrain_epoch_10.pth'
        if encoder_path:
            path = encoder_path
        save_model = torch.load(path, weights_only=True)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
            
        self.decoder = CASCADE(channels=[512, 320, 128, 64])
        print('Model %s created, param count: %d' %
                     ('decoder: ', sum([m.numel() for m in self.decoder.parameters()])))
        self.num_fg = num_fg
        self.num_bg = num_bg
        self.num_total = num_fg + num_bg
        
        self.decoder_dim = 64 
        
        # 可學習的原型
        self.prototypes = nn.Parameter(torch.randn(self.num_total, self.decoder_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, x):
        # Encoder
        x1, x2, x3, x4 = self.backbone(x)
        
        outs = self.decoder(x4, [x3, x2, x1])
        pixel_embeddings = outs[-1] # (B, 64, H/4, W/4)
        
        # Prototype Matching
        # Normalize
        embeddings_norm = F.normalize(pixel_embeddings, p=2, dim=1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)
        
        # Cosine Similarity: (B, K, H', W')
        similarity_map = F.conv2d(embeddings_norm, prototypes_norm.unsqueeze(-1).unsqueeze(-1))
        
        # Generate Binary Logits
        fg_score, _ = torch.max(similarity_map[:, :self.num_fg], dim=1)
        bg_score, _ = torch.max(similarity_map[:, self.num_fg:], dim=1)
        # Stack to (B, 2, H', W')
        # Channel 0: BG, Channel 1: FG
        logits_small = torch.stack([bg_score, fg_score], dim=1)
        
        # Upsample to original size (H, W)
        logits_high = F.interpolate(logits_small, size=x.shape[-2:], mode='bilinear', align_corners=False)
        similarity_map_high = F.interpolate(similarity_map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        # 返回:
        # 1. 二元 Logits (用於 Structure Loss)
        # 2. K類相似度圖 (用於 Sublabel Loss / Reg Loss / Visualization)
        return logits_high, similarity_map_high
        
if __name__ == '__main__':
    model = Prototype_CASCADE().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    logits_high, similarity_map_high = model(input_tensor)
    print(logits_high.size(), similarity_map_high.size())
