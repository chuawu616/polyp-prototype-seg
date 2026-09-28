import pytest
import torch

from conftest import DINO_CONFIGS, PVT_CONFIGS, RUN_DINOV3, offline_config
from protoseg_polyp.engine.losses import Criterion
from protoseg_polyp.models import build_model
from protoseg_polyp.models.encoders import PVTEMCAD
from protoseg_polyp.ops import l2_normalize


def _batch(sp=False, size=64):
    x = torch.randn(2, 3, size, size)
    y = torch.zeros(2, 1, size, size)
    y[:, :, size // 4:3 * size // 4, size // 5:4 * size // 5] = 1
    s = torch.randint(0, 30, (2, size, size)) if sp else torch.full((2, size, size), -1)
    return x, y, s


def _train_step(cfg):
    model = build_model(cfg['model']).train()
    x, y, s = _batch(sp=bool(cfg['data'].get('superpixel_root')))
    out = model(x, gt=y.squeeze(1).long(), sp=s, alpha=0.5)
    loss, terms = Criterion(cfg['loss'], model.head)(out, y, epoch=8)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    return model, out, loss, terms, grads


@pytest.mark.parametrize('name', PVT_CONFIGS)
def test_config_trains_one_step(name):
    cfg = offline_config(name)
    model, out, loss, terms, grads = _train_step(cfg)
    assert torch.isfinite(loss)
    assert set(terms) == {k for k in cfg['loss']}          # every configured loss is active
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert out['fg_prob'].shape == (2, 1, 64, 64)
    model.eval()
    with torch.no_grad():
        p = model(_batch()[0])['fg_prob']
    assert p.shape == (2, 1, 64, 64) and 0 <= p.min() and p.max() <= 1


@pytest.mark.parametrize('name', [n for n in PVT_CONFIGS if n.startswith(('nonlearnable', 'pseudo'))])
def test_nonlearnable_prototypes_move_by_ema_only(name):
    cfg = offline_config(name)
    model = build_model(cfg['model'])
    assert 'prototypes' not in dict(model.head.named_parameters())   # not optimised by gradients
    # every forward re-normalises the prototypes (as the original code did); only training moves them
    before = l2_normalize(model.head.prototypes.clone())
    model.eval()
    with torch.no_grad():
        model(_batch()[0])
    assert torch.allclose(model.head.prototypes, before, atol=1e-6)  # eval: normalisation only
    model.train()
    x, y, s = _batch()
    model(x, gt=y.squeeze(1).long(), sp=s)
    assert not torch.allclose(model.head.prototypes, before)          # training step does (EMA)
    assert torch.allclose(model.head.prototypes.norm(dim=-1), torch.ones(before.shape[:2]), atol=1e-5)


def test_pvt_emcad_feature_nodes():
    enc = PVTEMCAD().eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        f = enc(x)
    shapes = {k: tuple(v.shape[1:]) for k, v in f.items()}
    assert shapes['d1'] == (64, 16, 16) and shapes['dd4'] == (512, 2, 2)
    assert shapes['dd3'][0] == 320 and shapes['dd2'][0] == 128 and shapes['dd1'][0] == 64
    assert torch.equal(f['dd4'], f['x4'])
    assert not f['dd3'].requires_grad


def test_detach_taps_controls_gradient_through_dd4():
    for detach, expect_grad in ((True, False), (False, True)):
        enc = PVTEMCAD(detach_taps=detach)
        enc(torch.randn(1, 3, 64, 64))['dd4'].sum().backward() if not detach else None
        g = enc.backbone.patch_embed1.proj.weight.grad
        assert (g is not None and g.abs().sum() > 0) == expect_grad


@pytest.mark.skipif(not RUN_DINOV3, reason='set PROTOSEG_TEST_DINOV3=1 (downloads DINOv3 code via torch.hub)')
@pytest.mark.parametrize('name', DINO_CONFIGS)
def test_dinov3_config_trains_one_step(name):
    _, out, loss, _, _ = _train_step(offline_config(name))
    assert torch.isfinite(loss) and out['fg_prob'].shape == (2, 1, 64, 64)
