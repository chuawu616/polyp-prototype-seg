import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange, repeat
from timm.layers import trunc_normal_

from lib.pvtv2 import pvt_v2_b2
from lib.decoders import CASCADE
from lib.contrast import momentum_update, l2_normalize
from lib.sinkhorn import distributed_sinkhorn


class Prototype_Pseudo(nn.Module):
    """
    Prototype + Feature Map Dynamic Pseudo-label Segmentation

    核心設計：
    - Prototype（EMA 更新，non-learnable）作為 momentum teacher
    - Sinkhorn balanced assignment 產生 dynamic pseudo-label
    - Learnable seg_head（student）學習預測 pseudo-label（K*M 類）
    - CE loss：multi-class pseudo-label supervision → decoder 被迫保留 intra-class variation
    - Dice loss：binary GT supervision → 維持 FG/BG 分割精度

    相較於 v0：
    - 移除 PPC/PPD loss（CE loss 直接承擔 subclass 約束責任）
    - 新增 seg_head（learnable 1×1 conv，outputs K*M classes）
    - 分割預測來自 seg_head，而非 sim_matrix 的 amax
    - Binary prediction 由 K*M softmax 機率加總推導，確保一致性
    """

    def __init__(self, num_classes=2, num_prototype=5, gamma=0.999,
                 use_prototype=True, update_prototype=True,
                 encoder_path=None, pretrained_model_path=None,
                 kmeans_center_path=None):
        super(Prototype_Pseudo, self).__init__()

        self.num_classes = num_classes
        self.num_prototype = num_prototype
        self.gamma = gamma
        self.use_prototype = use_prototype
        self.update_prototype = update_prototype

        # ── Backbone ──────────────────────────────────────────────────────
        self.backbone = pvt_v2_b2()

        # ImageNet pretrained backbone（pretrained_model_path 未指定時載入）
        if pretrained_model_path is None:
            path = '/home/U116med/wch_code/non_learnable/weights/pvt_v2_b2.pth'
            if encoder_path:
                path = encoder_path
            save_model = torch.load(path, weights_only=True)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items()
                          if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)

        # ── Decoder ───────────────────────────────────────────────────────
        self.decoder = CASCADE(channels=[512, 320, 128, 64])

        # ── Prototype（d1 特徵空間，64-dim）──────────────────────────────
        self.in_channels = 64

        self.prototypes = nn.Parameter(
            torch.zeros(self.num_classes, self.num_prototype, self.in_channels),
            requires_grad=True
        )

        # Prototype 初始化：KMeans center 優先，否則 trunc_normal_
        if kmeans_center_path is not None:
            self._load_kmeans_centers(kmeans_center_path)
        else:
            trunc_normal_(self.prototypes, std=0.02)

        self.feat_norm = nn.LayerNorm(self.in_channels)

        # ── Segmentation Head（student）───────────────────────────────────
        # 輸出 K*M subclass logits
        # channel 排列：[BG_0,...,BG_{M-1}, FG_0,...,FG_{M-1}]
        self.seg_head = nn.Conv2d(
            self.in_channels,
            self.num_classes * self.num_prototype,
            kernel_size=1
        )
        nn.init.xavier_uniform_(self.seg_head.weight)
        nn.init.constant_(self.seg_head.bias, 0)

        # ── Full pretrained model 載入（最後執行，覆蓋以上所有初始化）──
        if pretrained_model_path is not None:
            self._load_pretrained_model(pretrained_model_path)

    def _load_pretrained_model(self, path):
        """
        載入完整 pretrained model weight（shape 相符的 key 才載入）。
        seg_head 等新模組在 pretrained 中不存在時自動跳過。
        """
        save_model = torch.load(path, map_location='cpu')
        model_dict = self.state_dict()
        state_dict = {
            k: v for k, v in save_model.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        print(f'[Prototype_Pseudo] Loaded {len(state_dict)} layers '
              f'from pretrained model: {path}')

    def _load_kmeans_centers(self, path):
        """
        載入 KMeans center 作為 prototype 初始值。
        shape 須符合 (num_classes, num_prototype, in_channels)。
        shape 不符時退回 trunc_normal_ 初始化。
        """
        import os
        if os.path.exists(path):
            kmeans_centers = torch.load(path, map_location='cpu')
            if kmeans_centers.shape == self.prototypes.shape:
                self.prototypes.data.copy_(kmeans_centers)
                print(f'[Prototype_Pseudo] Loaded KMeans centers from {path}.')
            else:
                print(f'[Prototype_Pseudo] Shape mismatch: '
                      f'expected {self.prototypes.shape}, '
                      f'got {kmeans_centers.shape}. Using trunc_normal_.')
                trunc_normal_(self.prototypes, std=0.02)
        else:
            print(f'[Prototype_Pseudo] KMeans file not found: {path}. '
                  f'Using trunc_normal_.')
            trunc_normal_(self.prototypes, std=0.02)

    def _binary_from_multiclass(self, pred_multiclass):
        """
        從 K*M class logits 推導 binary (FG/BG) 機率。

        使用 logsumexp 在 log-probability 空間做邊際化，
        保留 K*M 結構對 binary 預測的一致性。

        Args:
            pred_multiclass: (B, K*M, H, W) raw logits
        Returns:
            pred_binary: (B, 2, H, W) FG/BG 機率（已 softmax）
        """
        M = self.num_prototype
        # log P(class k) ∝ logsumexp over subclass logits within class k
        bg_logit = torch.logsumexp(pred_multiclass[:, :M,  :, :], dim=1, keepdim=True)
        fg_logit = torch.logsumexp(pred_multiclass[:, M:2*M, :, :], dim=1, keepdim=True)
        binary_logits = torch.cat([bg_logit, fg_logit], dim=1)  # (B, 2, H, W)
        return F.softmax(binary_logits, dim=1)                   # (B, 2, H, W) 機率
        
    @torch.no_grad()
    def prototype_learning(self, _c_d1, pred_binary_d1, d1_gt_label,
                           sim_matrix, sp_d1=None):
        """
        Prototype EMA + Sinkhorn pseudo-label。

        sp_d1 提供時（superpixel 模式）：
          Sinkhorn 在 superpixel 均值特徵上運算，同一 superpixel 內所有
          pixel 強制分到同一 prototype，天然保證空間連續性，計算量更小。
        sp_d1=None 時退回原始 pixel-level Sinkhorn。

        Args:
            _c_d1:          (B*H*W, C)   d1 特徵，已 l2_normalize
            pred_binary_d1: (B, 2, H, W) binary prediction（valid_mask 用）
            d1_gt_label:    (B*H*W,)     d1 解析度 GT class label
            sim_matrix:     (B*H*W,M,K)  特徵與 prototype cosine 相似度
            sp_d1:          (B*H*W,)|None d1 解析度 superpixel index
                            -1 = 無效（未提供 superpixel）
        Returns:
            pseudo_labels:  (B*H*W,)     Sinkhorn subclass label（K*M classes）
        """
        pred_seg   = torch.max(pred_binary_d1, 1)[1].view(-1)
        valid_mask = (d1_gt_label == pred_seg)

        pseudo_labels = d1_gt_label.clone().float()
        protos = self.prototypes.data.clone()
        C = _c_d1.shape[-1]

        use_sp = (sp_d1 is not None) and (sp_d1.min().item() >= 0)

        for k in range(self.num_classes):
            k_mask = (d1_gt_label == k)
            k_idx  = k_mask.nonzero(as_tuple=True)[0]
            if len(k_idx) == 0:
                continue

            if use_sp:
                # ── Superpixel-level Sinkhorn ──────────────────────────────
                # 1. class k 的 superpixel ID → remap 至連續 index
                sp_ids_k = sp_d1[k_idx]
                unique_sps, sp_remapped = torch.unique(
                    sp_ids_k, return_inverse=True
                )
                N_sp = len(unique_sps)

                # 2. Vectorized superpixel mean feature（scatter_add）
                features_k = _c_d1[k_idx]                  # (N_k, C)
                sp_sum = torch.zeros(N_sp, C,
                                     device=features_k.device,
                                     dtype=features_k.dtype)
                sp_cnt = torch.zeros(N_sp, 1,
                                     device=features_k.device,
                                     dtype=features_k.dtype)
                sp_sum.scatter_add_(
                    0, sp_remapped.unsqueeze(1).expand(-1, C), features_k
                )
                sp_cnt.scatter_add_(
                    0, sp_remapped.unsqueeze(1),
                    torch.ones(len(sp_remapped), 1, device=features_k.device)
                )
                sp_mean = F.normalize(
                    sp_sum / sp_cnt.clamp(min=1), p=2, dim=-1
                )                                           # (N_sp, C)

                # 3. Superpixel similarity → Sinkhorn
                init_q = torch.mm(sp_mean,
                                  l2_normalize(protos[k]).t())  # (N_sp, M)
                q, indexs = distributed_sinkhorn(init_q)        # (N_sp,M),(N_sp,)

                # 4. Propagate assignment 回 pixel level
                pixel_indexs = indexs[sp_remapped]          # (N_k,)
                pseudo_labels[k_idx] = (
                    pixel_indexs.float() + self.num_prototype * k
                )

                # 5. EMA：superpixel 傳回的 q propagate 到 pixel，再過 valid_mask
                q_pixels = q[sp_remapped]                   # (N_k, M)
                m_k      = valid_mask[k_idx]
                m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)
                m_q      = q_pixels * m_k_tile              # (N_k, M)
                c_k_tile = repeat(m_k, 'n -> n tile', tile=C)
                c_q      = features_k * c_k_tile            # (N_k, C)

            else:
                # ── Pixel-level Sinkhorn（fallback）───────────────────────
                init_q = sim_matrix[..., k][k_mask]         # (N_k, M)
                q, indexs = distributed_sinkhorn(init_q)

                pseudo_labels[k_mask] = (
                    indexs.float() + self.num_prototype * k
                )

                m_k      = valid_mask[k_mask]
                c_k      = _c_d1[k_mask]
                m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)
                m_q      = q * m_k_tile
                c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])
                c_q      = c_k * c_k_tile

            # EMA update（兩種模式共用）
            f = m_q.transpose(0, 1) @ c_q   # (M, C)
            n = torch.sum(m_q, dim=0)        # (M,)

            if torch.sum(n) > 0 and self.update_prototype:
                f = F.normalize(f, p=2, dim=-1)
                new_value = momentum_update(
                    old_value=protos[k, n != 0, :],
                    new_value=f[n != 0, :],
                    momentum=self.gamma
                )
                protos[k, n != 0, :] = new_value

        self.prototypes = nn.Parameter(l2_normalize(protos), requires_grad=False)

        if dist.is_available() and dist.is_initialized():
            protos = self.prototypes.data.clone()
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(protos, requires_grad=False)

        return pseudo_labels

    def forward(self, x, gt_semantic_seg=None, sp_map=None):
        # ── Encoder + Decoder ──────────────────────────────────────────────
        x1, x2, x3, x4 = self.backbone(x)
        outs = self.decoder(x4, [x3, x2, x1])
        d1_feat = outs[4]   # (B, 64, H/4, W/4)

        b, c, h, w = d1_feat.shape

        # ── Learnable Segmentation（Student）─────────────────────────────
        pred_multiclass = self.seg_head(d1_feat)           # (B, K*M, H, W)
        pred_binary     = self._binary_from_multiclass(pred_multiclass)

        pred_multiclass_high = F.interpolate(
            pred_multiclass, size=x.shape[-2:], mode='bilinear', align_corners=False
        )
        pred_binary_high = F.interpolate(
            pred_binary, size=x.shape[-2:], mode='bilinear', align_corners=False
        )

        # ── Prototype Teacher（EMA + Sinkhorn）────────────────────────────
        self.prototypes.data.copy_(l2_normalize(self.prototypes))

        _c_d1 = rearrange(d1_feat, 'b c h w -> (b h w) c')
        _c_d1 = self.feat_norm(_c_d1)
        _c_d1 = l2_normalize(_c_d1)

        sim_matrix = torch.einsum('nd,kmd->nmk', _c_d1, self.prototypes)

        sim_map_small = rearrange(sim_matrix, '(b h w) m k -> b (k m) h w', b=b, h=h)
        similarity_map_high = F.interpolate(
            sim_map_small, size=x.shape[-2:], mode='bilinear', align_corners=False
        )

        if self.training and self.use_prototype and gt_semantic_seg is not None:
            d1_gt_label = F.interpolate(
                gt_semantic_seg.unsqueeze(1).float(),
                size=(h, w), mode='nearest'
            ).view(-1)

            # Superpixel map 降採樣至 d1 解析度（NEAREST 保留整數 index）
            if sp_map is not None and sp_map.min().item() >= 0:
                sp_d1 = F.interpolate(
                    sp_map.unsqueeze(1).float(),
                    size=(h, w), mode='nearest'
                ).squeeze(1).long().view(-1)    # (B*H_d1*W_d1,)
            else:
                sp_d1 = None

            pseudo_labels = self.prototype_learning(
                _c_d1=_c_d1,
                pred_binary_d1=pred_binary,
                d1_gt_label=d1_gt_label,
                sim_matrix=sim_matrix,
                sp_d1=sp_d1
            )

            return {
                'seg':           pred_multiclass_high,
                'seg_binary':    pred_binary_high,
                'pseudo_labels': pseudo_labels,
                'similarity_map': similarity_map_high
            }

        return pred_binary_high, similarity_map_high


if __name__ == '__main__':
    model = Prototype_Pseudo(num_classes=2, num_prototype=5,
                             pretrained_model_path='SKIP').cuda()
    dummy_input = torch.randn(2, 3, 352, 352).cuda()
    dummy_gt    = torch.randint(0, 2, (2, 352, 352)).cuda()
    dummy_sp    = torch.randint(0, 200, (2, 352, 352)).cuda()  # superpixel map

    model.train()
    # Superpixel mode
    out = model(dummy_input, dummy_gt, sp_map=dummy_sp)
    print("Train (SP) | seg:          ", out['seg'].shape)
    print("Train (SP) | seg_binary:   ", out['seg_binary'].shape)
    print("Train (SP) | pseudo_labels:", out['pseudo_labels'].shape)

    # Pixel-level fallback
    out2 = model(dummy_input, dummy_gt, sp_map=None)
    print("Train (px) | pseudo_labels:", out2['pseudo_labels'].shape)

    model.eval()
    binary, sim = model(dummy_input)
    print("Eval       | binary:       ", binary.shape)
    print("Eval       | sim_map:      ", sim.shape)