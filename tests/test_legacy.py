"""Checkpoints of the original research code (flat module names) must load into the refactored models."""
import pytest
import torch

from conftest import offline_config
from protoseg_polyp.models import build_model, convert_legacy_state_dict, load_checkpoint

TO_LEGACY = [('encoder.backbone.', 'backbone.'), ('encoder.decoder.', 'decoder.'), ('neck.', 'proj.')]


def _as_legacy(sd, head_out='out_head4.'):
    """Invert the conversion: produce the key names the original code used."""
    out = {}
    for k, v in sd.items():
        for new, old in TO_LEGACY:
            if k.startswith(new):
                out[old + k[len(new):]] = v
                break
        else:
            k = k[len('head.'):]
            out[(head_out + k[len('out.'):]) if k.startswith('out.') else k] = v
    return out


@pytest.mark.parametrize('name,head_out', [('pvt_linear.yaml', 'out_head4.'), ('pvt_linear.yaml', 'temp_head.'),
                                           ('pvt_proto_fb88.yaml', None),
                                           ('pvt_specialize_hybrid_fhb424_warm10.yaml', None),
                                           ('nonlearnable_v3_dd4.yaml', None), ('pseudo_fb33.yaml', None)])
def test_legacy_checkpoint_roundtrip(tmp_path, name, head_out):
    cfg = offline_config(name)
    src = build_model(cfg['model'])
    legacy = _as_legacy(src.state_dict(), head_out or 'out_head4.')
    if head_out:  # the linear baseline also stored an unused grayscale->RGB adapter
        legacy['conv.0.weight'] = torch.zeros(3, 1, 1, 1)
    path = tmp_path / 'legacy.pth'
    torch.save(legacy, path)
    dst = build_model(cfg['model'])
    load_checkpoint(dst, str(path))                  # auto-detects the legacy format, strict=True
    for k, v in src.state_dict().items():
        assert torch.equal(v, dst.state_dict()[k]), k


def test_dinov3_concat_conv_head_mapping():
    sd = {'head.weight': torch.zeros(1, 256, 1, 1), 'head.bias': torch.zeros(1), 'head.0.weight': torch.zeros(2, 2),
          'proj.0.weight': torch.zeros(1), 'proto_proj.1.bias': torch.zeros(1), 'backbone.blocks.0.x': torch.zeros(1)}
    out = convert_legacy_state_dict(sd)
    assert set(out) == {'head.out.weight', 'head.out.bias', 'head.head.0.weight', 'neck.0.weight', 'neck.1.bias',
                        'encoder.backbone.blocks.0.x'}


def test_shape_matched_initialisation(tmp_path):
    """init_from: a linear-baseline checkpoint initialises encoder + decoder of a prototype model."""
    lin = build_model(offline_config('pvt_linear.yaml')['model'])
    path = tmp_path / 'lin.pth'
    torch.save(lin.state_dict(), path)
    proto = build_model(offline_config('pseudo_fb33.yaml')['model'])
    load_checkpoint(proto, str(path), shape_match=True)
    k = 'encoder.decoder.mscb1.0.pconv1.0.weight'
    assert torch.equal(proto.state_dict()[k], lin.state_dict()[k])
