"""Learnable prediction heads.

Every head maps an encoder feature map to a dict that always contains
    fg_logit : (B, 1, H, W) at input resolution, sigmoid(fg_logit) = foreground probability
and, for prototype heads,
    scores   : (B, 2, H, W) per-class score [BG, FG] before the FG-BG contrast
    sim      : (B, K, H, W) similarity to every prototype (for visualisation / utilisation analysis)
Parameter names follow the original research code so old checkpoints load after a prefix rename.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _up(t, size):
    return F.interpolate(t, size=size, mode='bilinear', align_corners=False)


def _cos(feat, protos):
    """Cosine similarity between a (B,C,H,W) map and (K,C) prototypes -> (B,K,H,W)."""
    return F.conv2d(F.normalize(feat, p=2, dim=1), F.normalize(protos, p=2, dim=1)[..., None, None])


class LinearHead(nn.Module):
    """1x1 conv to one logit (EMCAD baseline)."""

    def __init__(self, in_dim):
        super().__init__()
        self.out = nn.Conv2d(in_dim, 1, 1)

    def forward(self, feat, size, **_):
        return dict(fg_logit=_up(self.out(feat), size))


class MLPHead(nn.Module):
    """Per-pixel MLP used on top of frozen / fine-tuned DINOv3 features."""

    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, feat, size, **_):
        b, c, h, w = feat.shape
        logit = self.head(feat.permute(0, 2, 3, 1).reshape(-1, c)).view(b, h, w, 1).permute(0, 3, 1, 2)
        return dict(fg_logit=_up(logit, size))


class PrototypeHead(nn.Module):
    """Learnable prototypes replace the linear head (the author's baseline).

    score_c = max_k cos(f, p_{c,k});  fg_logit = (score_fg - score_bg) * 10
    `prototypes` rows are ordered [FG_0..FG_{F-1}, BG_0..BG_{B-1}].
    """

    def __init__(self, in_dim, num_fg=8, num_bg=8, scale=10.0):
        super().__init__()
        self.num_fg, self.num_bg, self.scale = num_fg, num_bg, scale
        self.prototypes = nn.Parameter(torch.randn(num_fg + num_bg, in_dim))
        nn.init.xavier_uniform_(self.prototypes)
        self.fg_channels = list(range(num_fg))
        self.bg_channels = list(range(num_fg, num_fg + num_bg))

    def forward(self, feat, size, **_):
        sim = _cos(feat, self.prototypes)
        fg = sim[:, :self.num_fg].amax(dim=1)
        bg = sim[:, self.num_fg:].amax(dim=1)
        scores = _up(torch.stack([bg, fg], dim=1), size)
        return dict(scores=scores, sim=_up(sim, size),
                    fg_logit=(scores[:, 1:2] - scores[:, 0:1]) * self.scale)


class DynamicHardPrototypes(nn.Module):
    """Hybrid variant: hard prototypes = global parameters + image-conditioned attention pooling."""

    def __init__(self, num_hard, in_dim):
        super().__init__()
        self.global_hard = nn.Parameter(torch.randn(num_hard, in_dim))
        nn.init.xavier_uniform_(self.global_hard)
        self.dynamic_extractor = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, 3, padding=1, bias=False), nn.BatchNorm2d(in_dim // 2),
            nn.ReLU(inplace=True), nn.Conv2d(in_dim // 2, num_hard, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat, alpha=1.0):
        b, c, h, w = feat.shape
        attn = F.softmax(self.dynamic_extractor(feat).view(b, -1, h * w), dim=-1)  # (B, H, HW)
        dynamic = torch.bmm(attn, feat.view(b, c, -1).transpose(1, 2))           # (B, H, C)
        return self.global_hard.unsqueeze(0).expand(b, -1, -1) + self.gamma * dynamic * alpha


class SpecializeHead(nn.Module):
    """Easy / hard foreground prototypes. The hard set only corrects pixels the easy set is unsure of:

        s_fg = s_easy + (1 - sigmoid(s_easy - s_bg)) * alpha * s_hard

    with s_* = max-cosine x learnable logit scale and alpha warmed up 0 -> 1 during training.
    `num_fg` counts easy + hard (fhb424 = 4 FG of which 2 hard, 4 BG).
    """

    def __init__(self, in_dim, num_fg=4, num_hard=2, num_bg=4, hybrid=False, scale=10.0):
        super().__init__()
        assert num_fg >= num_hard + 1
        self.num_easy, self.num_hard, self.num_bg, self.hybrid, self.scale = \
            num_fg - num_hard, num_hard, num_bg, hybrid, scale
        self.easy_prototypes = nn.Parameter(torch.randn(self.num_easy, in_dim))
        if hybrid:
            self.hard_prototypes_module = DynamicHardPrototypes(num_hard, in_dim)
        else:
            self.hard_prototypes = nn.Parameter(torch.randn(num_hard, in_dim))
        self.bg_prototypes = nn.Parameter(torch.randn(num_bg, in_dim))
        for p in (self.easy_prototypes, self.bg_prototypes) + (() if hybrid else (self.hard_prototypes,)):
            nn.init.xavier_uniform_(p)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.fg_channels = list(range(num_fg))
        self.bg_channels = list(range(num_fg, num_fg + num_bg))

    def forward(self, feat, size, alpha=1.0, **_):
        scale = torch.clamp(self.logit_scale.exp(), max=100)
        sim_easy = _cos(feat, self.easy_prototypes) * scale
        sim_bg = _cos(feat, self.bg_prototypes) * scale
        if self.hybrid:
            hard = F.normalize(self.hard_prototypes_module(feat, alpha), p=2, dim=-1)
            sim_hard = torch.einsum('bchw,bnc->bnhw', F.normalize(feat, p=2, dim=1), hard) * scale
        else:
            sim_hard = _cos(feat, self.hard_prototypes) * scale
        s_easy, s_hard, s_bg = sim_easy.amax(1), sim_hard.amax(1), sim_bg.amax(1)
        s_fg = s_easy + (1.0 - torch.sigmoid(s_easy - s_bg)) * alpha * s_hard
        scores = _up(torch.stack([s_bg, s_fg], dim=1), size)
        return dict(scores=scores, sim=_up(torch.cat([sim_easy, sim_hard, sim_bg], 1), size),
                    fg_logit=(scores[:, 1:2] - scores[:, 0:1]) * self.scale)

    def prototype_groups(self):
        """(easy, hard) prototype tensors for the orthogonality loss."""
        return self.easy_prototypes, self.hard_prototypes
