"""Non-learnable prototype heads (Sinkhorn assignment + EMA update), Stage 5 of the project.

Prototypes are buffers of shape (K classes, M sub-prototypes, C). They receive no gradient: after
every training forward pass, pixels of class k are assigned to its M sub-prototypes by Sinkhorn and
each prototype moves (EMA) toward the mean of its assigned, *correctly predicted* pixels.

Class order is k = 0 (background), 1 (foreground); sub-class channel index = k * M + m.

    V2      prototypes on d1; out = LayerNorm(max_m cos)                                (ProtoSeg-faithful)
    V3      "dual level": prototypes live on a deep node (dd4), an MLP projects them to d1
    V0      everything on d1, no pre-training / KMeans
    Pseudo  prototypes are a teacher: Sinkhorn assignments become K*M-way pseudo-labels
            for a learnable 1x1-conv student; superpixel-level Sinkhorn optional
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import trunc_normal_

from ..ops import distributed_sinkhorn, l2_normalize, momentum_update


def _up(t, size):
    return F.interpolate(t, size=size, mode='bilinear', align_corners=False)


def _nearest(label, hw):
    """(B,H,W) integer map -> flattened (B*h*w,) float map at resolution hw."""
    return F.interpolate(label.unsqueeze(1).float(), size=hw, mode='nearest').view(-1)


class _NonLearnableBase(nn.Module):
    needs_all_feats = True

    def __init__(self, num_prototype, dim, gamma=0.999, update_prototype=True, kmeans_centers=None):
        super().__init__()
        self.num_classes, self.num_prototype, self.gamma = 2, num_prototype, gamma
        self.update_prototype = update_prototype
        protos = torch.zeros(self.num_classes, num_prototype, dim)
        trunc_normal_(protos, std=0.02)
        if kmeans_centers and os.path.exists(kmeans_centers):
            centers = torch.load(kmeans_centers, map_location='cpu')
            if centers.shape == protos.shape:
                protos = centers.clone()
        self.register_buffer('prototypes', protos)
        M = num_prototype
        self.bg_channels, self.fg_channels = list(range(M)), list(range(M, 2 * M))

    def _similarity(self, c):
        """c: (N, C) normalized features -> (N, M, K) cosine similarity."""
        self.prototypes.copy_(l2_normalize(self.prototypes))
        return torch.einsum('nd,kmd->nmk', c, self.prototypes)

    @torch.no_grad()
    def _ema(self, protos, k, q, valid, feats):
        """Move the class-k prototypes toward the mean of their assigned, valid pixels."""
        m_q = q * valid[:, None].float()
        c_q = feats * valid[:, None].float()
        f = m_q.t() @ c_q                 # (M, C)
        n = m_q.sum(dim=0)                # (M,)
        if n.sum() > 0 and self.update_prototype:
            f = F.normalize(f, p=2, dim=-1)
            protos[k, n != 0] = momentum_update(protos[k, n != 0], f[n != 0], self.gamma)

    @torch.no_grad()
    def _assign(self, feats, labels, sim, valid):
        """Sinkhorn per class + EMA. Returns sub-class labels (N,) in [0, K*M)."""
        sub_labels = labels.clone()
        protos = self.prototypes.clone()
        for k in range(self.num_classes):
            sel = labels == k
            if sel.sum() == 0:
                continue
            q, idx = distributed_sinkhorn(sim[..., k][sel])
            self._ema(protos, k, q, valid[sel], feats[sel])
            sub_labels[sel] = idx.float() + self.num_prototype * k
        # rebind instead of copy_: the old tensor may still be referenced by this step's autograd graph
        self.prototypes = l2_normalize(protos)
        return sub_labels


class NonLearnableV2Head(_NonLearnableBase):
    """ProtoSeg ported to d1: out_seg = LayerNorm_K(max_m cos). Contrast logits are raw cosine."""

    def __init__(self, in_dim=64, num_prototype=5, **kw):
        super().__init__(num_prototype, in_dim, **kw)
        self.feat_norm = nn.LayerNorm(in_dim)
        self.mask_norm = nn.LayerNorm(self.num_classes)

    def forward(self, feats, size, gt=None, **_):
        d1 = feats['d1']
        b, _, h, w = d1.shape
        c = l2_normalize(self.feat_norm(rearrange(d1, 'b c h w -> (b h w) c')))
        sim = self._similarity(c)
        out_seg = self.mask_norm(sim.amax(dim=1))
        scores = _up(rearrange(out_seg, '(b h w) k -> b k h w', b=b, h=h), size)
        out = dict(scores=scores, fg_logit=(scores[:, 1:2] - scores[:, 0:1]) * 10.0,
                   sim=_up(rearrange(sim, '(b h w) m k -> b (k m) h w', b=b, h=h), size))
        if self.training and gt is not None:
            labels = _nearest(gt, (h, w))
            valid = labels == out_seg.argmax(dim=1).float()
            out['contrast_logits'] = c @ self.prototypes.view(-1, self.prototypes.shape[-1]).t()
            out['contrast_target'] = self._assign(c, labels, sim, valid)
        return out


class NonLearnableV3Head(_NonLearnableBase):
    """Dual-level: true prototypes on `node` (default dd4, 11x11x512), projected by an MLP to d1 (64-d).

    PPC acts on the node level, PPD and the segmentation on d1. Node sub-class labels are
    nearest-upsampled to d1 (each dd4 cell becomes an 8x8 block).
    """
    NODE_DIMS = {'x1': 64, 'x2': 128, 'x3': 320, 'x4': 512, 'dd4': 512, 'dd3': 320, 'dd2': 128, 'dd1': 64}

    def __init__(self, in_dim=64, num_prototype=5, node='dd4', **kw):
        node_dim = self.NODE_DIMS[node]
        super().__init__(num_prototype, node_dim, **kw)
        self.node = node
        self.feat_norm_node = nn.LayerNorm(node_dim)
        self.feat_norm_d1 = nn.LayerNorm(in_dim)
        self.register_buffer('temperature', torch.tensor(10.0))
        self.proto_proj_mlp = nn.Sequential(nn.Linear(node_dim, node_dim), nn.LayerNorm(node_dim), nn.GELU(),
                                            nn.Linear(node_dim, in_dim))
        for m in self.proto_proj_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, feats, size, gt=None, **_):
        node, d1 = feats[self.node], feats['d1']
        bn, _, hn, wn = node.shape
        b, _, h, w = d1.shape
        c_node = l2_normalize(self.feat_norm_node(rearrange(node, 'b c h w -> (b h w) c')))
        c_d1 = l2_normalize(self.feat_norm_d1(rearrange(d1, 'b c h w -> (b h w) c')))
        sim_node = self._similarity(c_node)
        proj = l2_normalize(self.proto_proj_mlp(self.prototypes))            # (K, M, 64), learnable path
        sim_d1 = torch.einsum('nd,kmd->nmk', c_d1, proj)
        out_seg = rearrange(sim_d1.amax(dim=1), '(b h w) k -> b k h w', b=b, h=h)
        scores = _up(out_seg, size)
        out = dict(scores=scores, fg_logit=(scores[:, 1:2] - scores[:, 0:1]) * 10.0,
                   sim=_up(rearrange(sim_d1, '(b h w) m k -> b (k m) h w', b=b, h=h), size))
        if self.training and gt is not None:
            labels = _nearest(gt, (hn, wn))
            pred_node = F.interpolate(out_seg, size=(hn, wn), mode='bilinear', align_corners=False)
            valid = labels == pred_node.argmax(dim=1).view(-1).float()
            out['contrast_logits_node'] = (c_node @ self.prototypes.view(-1, self.prototypes.shape[-1]).t()) \
                * self.temperature
            out['contrast_logits'] = (c_d1 @ proj.view(-1, proj.shape[-1]).t()) * self.temperature
            target_node = self._assign(c_node, labels, sim_node, valid)
            out['contrast_target_node'] = target_node
            out['contrast_target'] = F.interpolate(target_node.view(bn, 1, hn, wn), size=(h, w),
                                                   mode='nearest').view(-1)
            out['temperature'] = self.temperature
        return out


class NonLearnableV0Head(_NonLearnableBase):
    """All prototype operations on d1; segmentation = max-cosine per class."""

    def __init__(self, in_dim=64, num_prototype=3, **kw):
        super().__init__(num_prototype, in_dim, **kw)
        self.feat_norm = nn.LayerNorm(in_dim)
        self.register_buffer('temperature', torch.tensor(10.0))

    def forward(self, feats, size, gt=None, **_):
        d1 = feats['d1']
        b, _, h, w = d1.shape
        c = l2_normalize(self.feat_norm(rearrange(d1, 'b c h w -> (b h w) c')))
        sim = self._similarity(c)
        out_seg = rearrange(sim.amax(dim=1), '(b h w) k -> b k h w', b=b, h=h)
        scores = _up(out_seg, size)
        out = dict(scores=scores, fg_logit=(scores[:, 1:2] - scores[:, 0:1]) * 10.0,
                   sim=_up(rearrange(sim, '(b h w) m k -> b (k m) h w', b=b, h=h), size))
        if self.training and gt is not None:
            labels = _nearest(gt, (h, w))
            valid = labels == out_seg.argmax(dim=1).view(-1).float()
            out['contrast_logits'] = (c @ self.prototypes.view(-1, self.prototypes.shape[-1]).t()) * self.temperature
            out['contrast_target'] = self._assign(c, labels, sim, valid)
            out['temperature'] = self.temperature
        return out


class PseudoLabelHead(_NonLearnableBase):
    """Prototypes as an EMA teacher producing K*M-way pseudo-labels for a learnable student.

    Student: 1x1 conv -> K*M logits [BG_0..BG_{M-1}, FG_0..FG_{M-1}].
    Binary probability: softmax over (logsumexp of BG logits, logsumexp of FG logits).
    With a superpixel map, Sinkhorn runs on superpixel mean features and every pixel of a
    superpixel inherits its assignment.
    """

    def __init__(self, in_dim=64, num_prototype=3, **kw):
        super().__init__(num_prototype, in_dim, **kw)
        self.feat_norm = nn.LayerNorm(in_dim)
        self.seg_head = nn.Conv2d(in_dim, self.num_classes * num_prototype, 1)
        nn.init.xavier_uniform_(self.seg_head.weight)
        nn.init.constant_(self.seg_head.bias, 0)

    def _binary(self, logits):
        M = self.num_prototype
        return F.softmax(torch.cat([torch.logsumexp(logits[:, :M], 1, keepdim=True),
                                    torch.logsumexp(logits[:, M:], 1, keepdim=True)], dim=1), dim=1)

    @torch.no_grad()
    def _assign_superpixel(self, feats, labels, sp, valid):
        sub_labels = labels.clone()
        protos = self.prototypes.clone()
        C = feats.shape[-1]
        for k in range(self.num_classes):
            idx = (labels == k).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                continue
            uniq, remap = torch.unique(sp[idx], return_inverse=True)
            fk = feats[idx]
            sp_sum = torch.zeros(len(uniq), C, device=fk.device, dtype=fk.dtype).scatter_add_(
                0, remap[:, None].expand(-1, C), fk)
            sp_cnt = torch.zeros(len(uniq), 1, device=fk.device, dtype=fk.dtype).scatter_add_(
                0, remap[:, None], torch.ones(len(remap), 1, device=fk.device))
            sp_mean = F.normalize(sp_sum / sp_cnt.clamp(min=1), p=2, dim=-1)
            q, sp_idx = distributed_sinkhorn(sp_mean @ l2_normalize(protos[k]).t())
            sub_labels[idx] = sp_idx[remap].float() + self.num_prototype * k
            self._ema(protos, k, q[remap], valid[idx], fk)
        # rebind instead of copy_: the old tensor may still be referenced by this step's autograd graph
        self.prototypes = l2_normalize(protos)
        return sub_labels

    def forward(self, feats, size, gt=None, sp=None, **_):
        d1 = feats['d1']
        b, _, h, w = d1.shape
        logits = self.seg_head(d1)
        binary = self._binary(logits)
        binary_up = _up(binary, size)
        c = l2_normalize(self.feat_norm(rearrange(d1, 'b c h w -> (b h w) c')))
        sim = self._similarity(c)
        out = dict(binary=binary_up, fg_prob=binary_up[:, 1:2], multiclass=_up(logits, size),
                   sim=_up(rearrange(sim, '(b h w) m k -> b (k m) h w', b=b, h=h), size))
        if self.training and gt is not None:
            labels = _nearest(gt, (h, w))
            valid = labels == binary.argmax(dim=1).view(-1).float()
            if sp is not None and sp.min() >= 0:
                sp_d1 = _nearest(sp, (h, w)).long()
                out['pseudo_labels'] = self._assign_superpixel(c, labels, sp_d1, valid)
            else:
                out['pseudo_labels'] = self._assign(c, labels, sim, valid)
            out['pseudo_hw'] = (h, w)
        return out
