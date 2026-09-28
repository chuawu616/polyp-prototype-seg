import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange, repeat
from timm.layers import trunc_normal_

from lib.pvtv2 import pvt_v2_b2
from lib.decoders import CASCADE
from lib.contrast import momentum_update, l2_normalize, ProjectionHead
from lib.sinkhorn import distributed_sinkhorn

class Prototype_CASCADE_v2(nn.Module):
    def __init__(self, num_classes=2, num_prototype=5, gamma=0.999, 
                 use_prototype=True, update_prototype=True, pretrain_prototype=False,
                 encoder_path=None, pretrained_model_path=None, kmeans_center_path=None):
        super(Prototype_CASCADE_v2, self).__init__()
        
        self.num_classes = num_classes
        self.num_prototype = num_prototype
        self.gamma = gamma
        self.use_prototype = use_prototype
        self.update_prototype = update_prototype
        self.pretrain_prototype = pretrain_prototype

        self.backbone = pvt_v2_b2()

        if pretrained_model_path is None:
            path = '/home/U116med/wch_code/non_learnable/weights/pvt_v2_b2.pth'
            if encoder_path:
                path = encoder_path

            save_model = torch.load(path, weights_only=True)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)
            
        self.decoder = CASCADE(channels=[512, 320, 128, 64])

        in_channels = 64

        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, in_channels),
                                       requires_grad=True)

        self.feat_norm = nn.LayerNorm(in_channels)
        self.mask_norm = nn.LayerNorm(self.num_classes)
        # self.temperature = nn.Parameter(torch.tensor(10.0), requires_grad=False)

        # 處理 Prototype 初始化
        if kmeans_center_path is not None:
            self._load_kmeans_centers(kmeans_center_path)
        else:
            trunc_normal_(self.prototypes, std=0.02)
        
        # 執行完整預訓練模型的權重轉移
        if pretrained_model_path is not None:
            self._load_pretrained_model(pretrained_model_path)

    def _load_pretrained_model(self, path):
        save_model = torch.load(path, map_location='cpu')
        model_dict = self.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys() and v.shape == model_dict[k].shape}
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        print(f"Loaded {len(state_dict)} shared layers from {path}.")

    def _load_kmeans_centers(self, path):
        if os.path.exists(path):
            kmeans_centers = torch.load(path, map_location='cpu')
            if kmeans_centers.shape == self.prototypes.shape:
                self.prototypes.data.copy_(kmeans_centers)
                print(f"Loaded K-Means centers from {path}.")
            else:
                print(f"Shape mismatch! Expected {self.prototypes.shape}, but got {kmeans_centers.shape}. Using trunc_normal_ instead.")
                trunc_normal_(self.prototypes, std=0.02)
        else:
            print(f"K-Means center file not found at {path}. Using trunc_normal_ instead.")
            trunc_normal_(self.prototypes, std=0.02)

    def prototype_learning(self, _c, out_seg, gt_seg, masks):
        """
        參數解析：
        _c: 降維並 L2 正規化後的像素特徵 (N, C_dim)，N = B*H*W
        out_seg: 當前模型預測出來的類別 Logits (N, num_classes)
        gt_seg: 降採樣攤平後的 Ground Truth 標籤 (N)
        masks: 像素特徵與「各類別、各個子原型」的相似度矩陣 (N, num_prototype, num_classes)
        """
        
        # =========================================================
        # 1. 建立「高可信度像素遮罩」 (Reliable Pixel Filtering)
        # 目的：不要用模型猜錯、或是位於邊界模糊的像素來更新 Prototype。
        # 只有當前預測類別 (pred_seg) 與真實類別 (gt_seg) 一致的像素，才具備更新資格。
        # =========================================================
        pred_seg = torch.max(out_seg, 1)[1]
        mask = (gt_seg == pred_seg.view(-1)) # boolean mask (N)

        # 計算像素特徵與「所有類別、所有子原型」的 Cosine 相似度
        # proto_logits shape: (N, num_classes * num_prototype)
        cosine_similarity = torch.mm(_c, self.prototypes.view(-1, self.prototypes.shape[-1]).t())

        proto_logits = cosine_similarity
        proto_target = gt_seg.clone().float() # 準備用來存放每個像素被分配到的「子原型 Index」

        # 複製當前的 Prototype 作為更新的基準
        protos = self.prototypes.data.clone()
        
        # =========================================================
        # 2. 針對每一個類別 (k) 獨立進行 Sinkhorn 聚類與更新
        # 以你的息肉專案為例：k=0 是背景，k=1 是息肉
        # =========================================================
        for k in range(self.num_classes):
            # 取出特徵與第 k 類的 K 個子原型的相似度 (N, num_prototype)
            init_q = masks[..., k] 
            
            # 過濾出「Ground Truth 真的是第 k 類」的像素 
            init_q = init_q[gt_seg == k, ...] 
            if init_q.shape[0] == 0:
                continue # 這個 Batch 中沒有這類的像素，跳過更新

            # [核心 1]：執行 Sinkhorn 演算法
            # q: 分配權重矩陣 (N_k, num_prototype)，這裡是 hard assignment (one-hot)
            # indexs: 每個像素被分配到的子原型 index (0 ~ num_prototype-1)
            q, indexs = distributed_sinkhorn(init_q)

            # --- 計算新的子原型特徵中心 ---
            # 必須把剛才的「高可信度遮罩 (mask)」加上去
            m_k = mask[gt_seg == k]  # (N_k)
            c_k = _c[gt_seg == k, ...] # (N_k, C_dim) 屬於這類的像素特徵

            # 將 mask 擴張以匹配矩陣維度，準備過濾
            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)
            m_q = q * m_k_tile  # (N_k, num_prototype) 只有可信度高的像素權重保留

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])
            c_q = c_k * c_k_tile  # (N_k, C_dim) 只有可信度高的特徵保留

            # 矩陣相乘：算出這 K 個子原型的特徵總和
            # m_q.T (num_prototype, N_k) x c_q (N_k, C_dim) -> f (num_prototype, C_dim)
            f = m_q.transpose(0, 1) @ c_q  
            
            # 統計每個子原型被分配到幾個「高可信度像素」
            n = torch.sum(m_q, dim=0) # (num_prototype)

            # =========================================================
            # 3. 動量更新 (EMA Update)
            # =========================================================
            if torch.sum(n) > 0 and self.update_prototype is True:
                # 再次 L2 正規化新的特徵中心
                f = F.normalize(f, p=2, dim=-1)

                # [核心 2]：呼叫 momentum_update
                # 只有被分配到像素的子原型 (n != 0) 才進行更新，避免更新成 0 向量
                new_value = momentum_update(old_value=protos[k, n != 0, :], 
                                            new_value=f[n != 0, :],
                                            momentum=self.gamma, debug=False)
                protos[k, n != 0, :] = new_value

            # 將這批像素的 Target 改寫為它們專屬的子類別 Index
            # 例如：如果 k=1 (息肉), K=5, 則子類的 index 為 5, 6, 7, 8, 9
            # 這會被送去計算 PPC 與 PPD Loss
            proto_target[gt_seg == k] = indexs.float() + (self.num_prototype * k)

        # 將更新後的 Prototypes 寫回模型的 Parameter/Buffer 中
        self.prototypes = nn.Parameter(l2_normalize(protos), requires_grad=False)

        # =========================================================
        # 4. 多 GPU 訓練的分散式同步 (Distributed All-Reduce)
        # 確保每張 GPU 算出來的局部 Prototype 被平均，維持全局統一
        # =========================================================
        if dist.is_available() and dist.is_initialized():
            protos = self.prototypes.data.clone()
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(protos, requires_grad=False)

        # 回傳這整批像素與所有 Prototype 的相似度，以及它們該對齊的 Target
        return proto_logits, proto_target

    def forward(self, x, gt_semantic_seg=None):
        # Backbone & Decoder 特徵擷取
        x1, x2, x3, x4 = self.backbone(x)
        outs = self.decoder(x4, [x3, x2, x1])
        decoder_feat = outs[-1] # (B, 64, H/4, W/4)

        b, _, h, w = decoder_feat.shape
        
        _c = rearrange(decoder_feat, 'b c h w -> (b h w) c')
        _c = self.feat_norm(_c)
        _c = l2_normalize(_c)

        self.prototypes.data.copy_(l2_normalize(self.prototypes))

        # n: h*w*b, k: num_class, m: num_prototype
        masks = torch.einsum('nd,kmd->nmk', _c, self.prototypes)

        out_seg = torch.amax(masks, dim=1)
        out_seg = self.mask_norm(out_seg)
        # out_set = out_seg * self.temperature
        out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=b, h=h)

        # 整理 similarity_map 以符合你的需求
        # 將 masks (N, M, K) 轉為 (B, K*M, H/4, W/4) 形式供外部視覺化與除錯
        sim_map_small = rearrange(masks, "(b h w) m k -> b (k m) h w", b=b, h=h)

        # 統一上採樣
        logits_high = F.interpolate(out_seg, size=x.shape[-2:], mode='bilinear', align_corners=False)
        similarity_map_high = F.interpolate(sim_map_small, size=x.shape[-2:], mode='bilinear', align_corners=False)

        if self.training and self.use_prototype is True and gt_semantic_seg is not None:
            gt_seg = F.interpolate(gt_semantic_seg.unsqueeze(1).float(), size=(h, w), mode='nearest').view(-1)
            contrast_logits, contrast_target = self.prototype_learning(_c, out_seg, gt_seg, masks)
                
            return {
                'seg': logits_high,                   
                'logits': contrast_logits,            
                'target': contrast_target,            
                'similarity_map': similarity_map_high 
            }

                
        # Inference 或無需更新 Prototype 時直接返回
        return logits_high, similarity_map_high

if __name__ == '__main__':
    model = Prototype_CASCADE_v2(num_classes=2, num_prototype=5).cuda()
    dummy_input = torch.randn(2, 3, 352, 352).cuda()
    dummy_gt = torch.randint(0, 2, (2, 1, 352, 352)).cuda()
    
    # 測試 Training 模式
    model.train()
    out_dict = model(dummy_input, dummy_gt)
    print("Training Seg Shape:", out_dict['seg'].shape)
    print("Training Sim Map Shape:", out_dict['similarity_map'].shape)
    
    # 測試 Inference 模式
    model.eval()
    logits, sim_map = model(dummy_input)
    print("Eval Logits Shape:", logits.shape)
    print("Eval Sim Map Shape:", sim_map.shape)