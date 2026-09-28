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

    def prototype_learning(self, _c_d1, pred_binary_d1, d1_gt_label, sim_matrix):
        """
        Prototype EMA 更新 + Sinkhorn pseudo-label 產生。

        Prototype（momentum teacher）：
          Sinkhorn balanced assignment → EMA 更新 prototype 位置

        Pseudo-label（student 的學習目標）：
          Sinkhorn assignment index → K*M class label
          Dynamic：隨 prototype 位置更新而改變

        Args:
            _c_d1:          (B*H*W, C)      d1 特徵，已 l2_normalize
            pred_binary_d1: (B, 2, H, W)    binary prediction（用於 valid_mask）
            d1_gt_label:    (B*H*W,)         d1 解析度 GT class label（0 or 1）
            sim_matrix:     (B*H*W, M, K)    特徵與 prototype 的 cosine 相似度
        Returns:
            pseudo_labels:  (B*H*W,)         Sinkhorn subclass label（K*M classes）
        """
        # Valid Pixel Mask：binary 預測正確的 pixel 才參與 EMA 更新
        pred_seg = torch.max(pred_binary_d1, 1)[1].view(-1)  # (B*H*W,)
        valid_mask = (d1_gt_label == pred_seg)

        pseudo_labels = d1_gt_label.clone().float()
        protos = self.prototypes.data.clone()

        for k in range(self.num_classes):
            init_q = sim_matrix[..., k]               # (B*H*W, M)
            init_q = init_q[d1_gt_label == k, ...]    # (N_k, M)
            if init_q.shape[0] == 0:
                continue

            # Sinkhorn：強制 balanced assignment，阻止 winner-takes-all
            q, indexs = distributed_sinkhorn(init_q)  # q: (N_k, M), indexs: (N_k,)

            # EMA 更新（只用 valid pixel）
            m_k = valid_mask[d1_gt_label == k]        # (N_k,) bool
            c_k = _c_d1[d1_gt_label == k, ...]        # (N_k, C)

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)
            m_q = q * m_k_tile                         # (N_k, M)

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])
            c_q = c_k * c_k_tile                       # (N_k, C)

            f = m_q.transpose(0, 1) @ c_q             # (M, C) weighted sum
            n = torch.sum(m_q, dim=0)                  # (M,)

            if torch.sum(n) > 0 and self.update_prototype:
                f = F.normalize(f, p=2, dim=-1)
                new_value = momentum_update(
                    old_value=protos[k, n != 0, :],
                    new_value=f[n != 0, :],
                    momentum=self.gamma
                )
                protos[k, n != 0, :] = new_value

            # Pseudo-label：Sinkhorn 分配的 subclass index
            pseudo_labels[d1_gt_label == k] = indexs.float() + (self.num_prototype * k)

        self.prototypes = nn.Parameter(l2_normalize(protos), requires_grad=False)

        if dist.is_available() and dist.is_initialized():
            protos = self.prototypes.data.clone()
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(protos, requires_grad=False)

        return pseudo_labels  # (B*H*W,)

    def forward(self, x, gt_semantic_seg=None):
        # ── Encoder + Decoder ──────────────────────────────────────────────
        x1, x2, x3, x4 = self.backbone(x)
        outs = self.decoder(x4, [x3, x2, x1])
        d1_feat = outs[4]   # (B, 64, H/4, W/4)，live gradient

        b, c, h, w = d1_feat.shape

        # ── Learnable Segmentation（Student）─────────────────────────────
        # seg_head：d1_feat → K*M class logits
        pred_multiclass = self.seg_head(d1_feat)           # (B, K*M, H, W)

        # Binary prediction：由 K*M 邊際化推導，保持一致性
        pred_binary = self._binary_from_multiclass(pred_multiclass)  # (B, 2, H, W)

        # 上採樣至原始解析度
        pred_multiclass_high = F.interpolate(
            pred_multiclass, size=x.shape[-2:], mode='bilinear', align_corners=False
        )  # (B, K*M, H_orig, W_orig)

        pred_binary_high = F.interpolate(
            pred_binary, size=x.shape[-2:], mode='bilinear', align_corners=False
        )  # (B, 2, H_orig, W_orig)

        # ── Prototype Teacher（EMA + Sinkhorn）────────────────────────────
        # 此分支只做 prototype 更新與 pseudo-label 產生，不參與 seg_head 的梯度
        self.prototypes.data.copy_(l2_normalize(self.prototypes))

        _c_d1 = rearrange(d1_feat, 'b c h w -> (b h w) c')
        _c_d1 = self.feat_norm(_c_d1)
        _c_d1 = l2_normalize(_c_d1)

        # Cosine 相似度矩陣：用於 EMA 更新和視覺化
        sim_matrix = torch.einsum('nd,kmd->nmk', _c_d1, self.prototypes)
        # (B*H*W, M, K)

        sim_map_small = rearrange(
            sim_matrix, '(b h w) m k -> b (k m) h w', b=b, h=h
        )
        similarity_map_high = F.interpolate(
            sim_map_small, size=x.shape[-2:], mode='bilinear', align_corners=False
        )

        if self.training and self.use_prototype and gt_semantic_seg is not None:
            # GT label 降採樣至 d1 解析度
            d1_gt_label = F.interpolate(
                gt_semantic_seg.unsqueeze(1).float(),
                size=(h, w), mode='nearest'
            ).view(-1)  # (B*H*W,)

            pseudo_labels = self.prototype_learning(
                _c_d1=_c_d1,
                pred_binary_d1=pred_binary,   # d1 解析度的 binary pred，用於 valid_mask
                d1_gt_label=d1_gt_label,
                sim_matrix=sim_matrix
            )

            return {
                # CE loss target：seg_head 輸出 vs pseudo_labels（K*M class）
                'seg':          pred_multiclass_high,  # (B, K*M, H, W) logits
                # Dice loss target：binary prediction vs binary GT
                'seg_binary':   pred_binary_high,      # (B, 2, H, W) 機率
                # Pseudo-label：Sinkhorn dynamic assignment（CE target）
                'pseudo_labels': pseudo_labels,         # (B*H*W,) long
                # Similarity map：prototype 相似度視覺化
                'similarity_map': similarity_map_high
            }

        # Eval：回傳 binary prediction
        return pred_binary_high, similarity_map_high


if __name__ == '__main__':
    # 測試一：不載入任何 pretrained weight（純隨機初始化）
    model = Prototype_Pseudo(num_classes=2, num_prototype=5,
                             pretrained_model_path='SKIP').cuda()
    dummy_input = torch.randn(2, 3, 352, 352).cuda()
    dummy_gt    = torch.randint(0, 2, (2, 352, 352)).cuda()

    model.train()
    out = model(dummy_input, dummy_gt)
    print("Train | seg (K*M logits):  ", out['seg'].shape)
    print("Train | seg_binary (prob): ", out['seg_binary'].shape)
    print("Train | pseudo_labels:     ", out['pseudo_labels'].shape)
    print("Train | similarity_map:    ", out['similarity_map'].shape)

    model.eval()
    binary, sim = model(dummy_input)
    print("Eval  | binary pred:       ", binary.shape)
    print("Eval  | similarity_map:    ", sim.shape)