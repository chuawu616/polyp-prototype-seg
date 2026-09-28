"""Segmentor = encoder -> (optional neck) -> head, built from a config dict."""
import torch
import torch.nn as nn

from .encoders import DINOv3, PVTEMCAD
from .heads import LinearHead, MLPHead, PrototypeHead, SpecializeHead
from .nonlearnable import NonLearnableV0Head, NonLearnableV2Head, NonLearnableV3Head, PseudoLabelHead

HEADS = {
    'linear': LinearHead,
    'mlp': MLPHead,
    'prototype': PrototypeHead,
    'specialize': SpecializeHead,
    'nonlearnable_v2': NonLearnableV2Head,
    'nonlearnable_v3': NonLearnableV3Head,
    'nonlearnable_v0': NonLearnableV0Head,
    'pseudo': PseudoLabelHead,
}


class Segmentor(nn.Module):
    def __init__(self, encoder, head, neck=None):
        super().__init__()
        self.encoder, self.neck, self.head = encoder, neck, head

    def forward(self, x, gt=None, sp=None, alpha=1.0):
        feats = self.encoder(x)
        key = self.encoder.out_key
        if self.neck is not None:
            feats[key] = self.neck(feats[key])
        inp = feats if getattr(self.head, 'needs_all_feats', False) else feats[key]
        out = self.head(inp, x.shape[-2:], gt=gt, sp=sp, alpha=alpha)
        if 'fg_prob' not in out:
            out['fg_prob'] = torch.sigmoid(out['fg_logit'])
        return out


def build_model(cfg):
    """cfg: {'encoder': {...}, 'neck': int | None, 'head': {...}} (see configs/*.yaml)."""
    enc_cfg = dict(cfg['encoder'])
    enc_type = enc_cfg.pop('type')
    if enc_type == 'pvt_emcad':
        encoder = PVTEMCAD(**enc_cfg)
    elif enc_type == 'dinov3':
        encoder = DINOv3(**enc_cfg)
    else:
        raise ValueError(enc_type)

    in_dim, neck = encoder.out_dim, None
    if cfg.get('neck'):
        neck = nn.Sequential(nn.Conv2d(in_dim, cfg['neck'], 1), nn.BatchNorm2d(cfg['neck']), nn.GELU())
        in_dim = cfg['neck']

    head_cfg = dict(cfg['head'])
    head = HEADS[head_cfg.pop('type')](in_dim=in_dim, **head_cfg)
    return Segmentor(encoder, head, neck)


# ----------------------------------------------------------------------------------------------
# Checkpoints trained with the original research code (one class per variant, flat module names)
# ----------------------------------------------------------------------------------------------
_LEGACY_PREFIX = [
    ('backbone.', 'encoder.backbone.'),
    ('decoder.', 'encoder.decoder.'),
    ('out_head4.', 'head.out.'),       # cascade/lib/networks.py
    ('temp_head.', 'head.out.'),       # non_learnable/lib/networks_pvtv2_cascade.py
    ('proto_proj.', 'neck.'),          # DINOv3 specialize
    ('proj.', 'neck.'),                # DINOv3 prototype
]


def convert_legacy_state_dict(sd):
    out = {}
    for k, v in sd.items():
        if k.startswith('conv.'):      # unused grayscale->RGB adapter of the linear baseline
            continue
        if k in ('head.weight', 'head.bias'):  # DINOv3_concat: 1x1 conv head
            out['head.out.' + k.split('.')[1]] = v
            continue
        for old, new in _LEGACY_PREFIX:
            if k.startswith(old):
                out[new + k[len(old):]] = v
                break
        else:
            out['head.' + k] = v        # every head parameter keeps its original name
    return out


def load_checkpoint(model, path, legacy=None, strict=True, shape_match=False):
    """Load a checkpoint. legacy=None auto-detects original-code checkpoints (no 'encoder.' keys).

    shape_match=True loads only keys whose shape matches (used to initialise a new head from a
    structure-loss-pretrained baseline, as the original `pretrained_model_path` option did).
    """
    sd = torch.load(path, map_location='cpu')
    if legacy is None:
        legacy = not any(k.startswith('encoder.') for k in sd)
    if legacy:
        sd = convert_legacy_state_dict(sd)
    if shape_match:
        own = model.state_dict()
        sd = {k: v for k, v in sd.items() if k in own and v.shape == own[k].shape}
        strict = False
    return model.load_state_dict(sd, strict=strict)
