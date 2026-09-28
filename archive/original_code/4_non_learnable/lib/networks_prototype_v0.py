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


class Prototype_CASCADE_v0(nn.Module):
    def __init__(self, num_classes=2, num_prototype=5, gamma=0.999,
                 use_prototype=True, update_prototype=True):
        super(Prototype_CASCADE_v0, self).__init__()

        self.num_classes = num_classes
        self.num_prototype = num_prototype
        self.gamma = gamma
        self.use_prototype = use_prototype
        self.update_prototype = update_prototype

        self.backbone = pvt_v2_b2()

        path = '/home/U116med/wch_code/non_learnable/weights/pvt_v2_b2.pth'

        save_model = torch.load(path, weights_only=True)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # Decoder
        self.decoder = CASCADE(channels=[512, 320, 128, 64])

        # Prototype 定義在 d1 特徵空間（64-dim）
        self.in_channels = 64

        self.prototypes = nn.Parameter(
            torch.zeros(self.num_classes, self.num_prototype, self.in_channels),
            requires_grad=True
        )
        trunc_normal_(self.prototypes, std=0.02)

        self.feat_norm = nn.LayerNorm(self.in_channels)
        self.temperature = nn.Parameter(torch.tensor(10.0), requires_grad=False)

    def prototype_learning(self, _c_d1, out_seg, d1_gt_label, sim_matrix, shape_d1):
        """
        所有 prototype 操作在 d1 層進行。
        Sinkhorn 在 d1 解析度直接產生 balanced assignment，
        避免 node_label nearest 上採樣帶來的解析度誤差。

        Args:
            _c_d1:       (B*H*W, C)  d1 特徵，已 l2_normalize
            out_seg:     (B, K, H, W)  當前分割預測（用於 valid_mask）
            d1_gt_label: (B*H*W,)    d1 解析度的 GT class label
            sim_matrix:  (B*H*W, M, K) d1 特徵與 prototype 的相似度矩陣
            shape_d1:    (B, H, W)
        Returns:
            contrast_logits: (B*H*W, K*M)  用於 PPC/PPD loss 的 logits
            proto_labels:    (B*H*W,)       Sinkhorn 分配的 subclass label
        """
        b, h, w = shape_d1

        # Valid Pixel Mask：d1 自己的分割預測與 GT 吻合的 pixel
        # 不依賴外部節點的預測，避免跨解析度的不一致
        pred_seg = torch.max(out_seg, 1)[1].view(-1)  # (B*H*W,)
        valid_mask = (d1_gt_label == pred_seg)

        # Contrast logits：每個 pixel 對所有 K*M 個 prototype 的相似度
        contrast_logits = torch.mm(
            _c_d1,
            self.prototypes.view(-1, self.in_channels).t()
        ) * self.temperature  # (B*H*W, K*M)

        proto_labels = d1_gt_label.clone().float()
        protos = self.prototypes.data.clone()

        for k in range(self.num_classes):
            # 取出 class k 的 pixel 對 M 個 prototype 的相似度
            init_q = sim_matrix[..., k]               # (B*H*W, M)
            init_q = init_q[d1_gt_label == k, ...]    # (N_k, M)
            if init_q.shape[0] == 0:
                continue

            # Sinkhorn：在 d1 解析度直接產生 balanced assignment
            # 強制每個 prototype 分到等量 pixel，阻止 winner-takes-all
            q, indexs = distributed_sinkhorn(init_q)  # q: (N_k, M), indexs: (N_k,)

            # Valid pixel 過濾：只用預測正確的 pixel 更新 prototype
            m_k = valid_mask[d1_gt_label == k]        # (N_k,) bool
            c_k = _c_d1[d1_gt_label == k, ...]        # (N_k, C)

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)
            m_q = q * m_k_tile                         # (N_k, M) masked assignment

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])
            c_q = c_k * c_k_tile                       # (N_k, C) masked features

            # EMA 更新：每個 prototype 移向分配到它的 valid pixel 的平均方向
            f = m_q.transpose(0, 1) @ c_q             # (M, C) weighted sum
            n = torch.sum(m_q, dim=0)                  # (M,) pixel count per prototype

            if torch.sum(n) > 0 and self.update_prototype:
                f = F.normalize(f, p=2, dim=-1)
                new_value = momentum_update(
                    old_value=protos[k, n != 0, :],
                    new_value=f[n != 0, :],
                    momentum=self.gamma
                )
                protos[k, n != 0, :] = new_value

            # Sinkhorn 分配的 subclass index 寫入 label
            proto_labels[d1_gt_label == k] = indexs.float() + (self.num_prototype * k)

        # Prototype 更新後重新 normalize
        self.prototypes = nn.Parameter(l2_normalize(protos), requires_grad=False)

        if dist.is_available() and dist.is_initialized():
            protos = self.prototypes.data.clone()
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(protos, requires_grad=False)

        return contrast_logits, proto_labels

    def forward(self, x, gt_semantic_seg=None):
        # Encoder
        x1, x2, x3, x4 = self.backbone(x)

        # Decoder：取 d1（最終輸出，64-dim，最高解析度）
        outs = self.decoder(x4, [x3, x2, x1])
        d1_feat = outs[4]  # (B, 64, H/4, W/4)

        b, c, h, w = d1_feat.shape

        # D1 特徵前處理
        _c_d1 = rearrange(d1_feat, 'b c h w -> (b h w) c')
        _c_d1 = self.feat_norm(_c_d1)
        _c_d1 = l2_normalize(_c_d1)

        # Prototype 正規化
        self.prototypes.data.copy_(l2_normalize(self.prototypes))

        # 相似度矩陣：d1 特徵 vs 所有 prototype
        # (B*H*W, M, K)：每個 pixel 對每個 class 的每個 prototype 的相似度
        sim_matrix = torch.einsum('nd,kmd->nmk', _c_d1, self.prototypes)

        # 分割預測：每個 class 取最大 prototype 相似度
        out_seg = torch.amax(sim_matrix, dim=1)                        # (B*H*W, K)
        out_seg = rearrange(out_seg, '(b h w) k -> b k h w', b=b, h=h)

        # Similarity map：所有 prototype 的相似度展開，用於視覺化
        sim_map_small = rearrange(
            sim_matrix, '(b h w) m k -> b (k m) h w', b=b, h=h
        )

        # 上採樣至原始輸入解析度
        logits_high = F.interpolate(
            out_seg, size=x.shape[-2:], mode='bilinear', align_corners=False
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

            contrast_logits, proto_labels = self.prototype_learning(
                _c_d1=_c_d1,
                out_seg=out_seg,
                d1_gt_label=d1_gt_label,
                sim_matrix=sim_matrix,
                shape_d1=(b, h, w)
            )

            return {
                'seg': logits_high,
                'logits': contrast_logits,   # (B*H*W, K*M)，用於 PPC/PPD loss
                'labels': proto_labels,       # (B*H*W,)，Sinkhorn subclass label
                'similarity_map': similarity_map_high
            }

        return logits_high, similarity_map_high


if __name__ == '__main__':
    model = Prototype_CASCADE_v0(num_classes=2, num_prototype=5).cuda()
    dummy_input = torch.randn(2, 3, 352, 352).cuda()
    dummy_gt = torch.randint(0, 2, (2, 352, 352)).cuda()

    model.train()
    out_dict = model(dummy_input, dummy_gt)
    print("Train | seg shape:         ", out_dict['seg'].shape)
    print("Train | logits shape:      ", out_dict['logits'].shape)
    print("Train | labels shape:      ", out_dict['labels'].shape)
    print("Train | similarity_map:    ", out_dict['similarity_map'].shape)

    model.eval()
    logits, sim_map = model(dummy_input)
    print("Eval  | logits shape:      ", logits.shape)
    print("Eval  | sim_map shape:     ", sim_map.shape)
