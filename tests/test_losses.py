import torch

from protoseg_polyp.engine.losses import (Criterion, dice_loss, kl_balance_loss, orthogonality_loss, ppd_loss,
                                          structure_loss)


def test_ppd_temperature_handling():
    cos = torch.tensor([[0.5, 0.1], [0.2, 0.9]])
    target = torch.tensor([0, 1])
    T = torch.tensor(10.0)
    fixed = ppd_loss(cos * T, target, temperature=T)
    legacy = ppd_loss(cos * T, target, temperature=T, legacy_scaled=True)
    assert torch.isclose(fixed, ((1 - torch.tensor([0.5, 0.9])) ** 2).mean())
    assert torch.isclose(legacy, ((1 - 10 * torch.tensor([0.5, 0.9])) ** 2).mean())   # original behaviour


def test_kl_balance_is_zero_for_uniform_usage():
    sim = torch.zeros(1, 4, 8, 8)                     # all prototypes equally similar
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., :4] = 1
    assert kl_balance_loss(sim, mask, [0, 1], [2, 3]).abs() < 1e-6
    sim[:, 0] = 5.0                                   # one FG prototype dominates
    assert kl_balance_loss(sim, mask, [0, 1], [2, 3]) > 0.1


def test_segmentation_losses_prefer_the_right_answer():
    mask = torch.zeros(1, 1, 32, 32)
    mask[..., 8:24, 8:24] = 1
    good, bad = (mask * 2 - 1) * 5, -(mask * 2 - 1) * 5
    assert structure_loss(good, mask) < structure_loss(bad, mask)
    assert dice_loss(torch.sigmoid(good), mask) < 0.05 < dice_loss(torch.sigmoid(bad), mask)


def test_orthogonality_and_kl_ramp():
    e = torch.eye(4)[:2]
    assert orthogonality_loss(e, torch.eye(4)[2:]) == 0
    w = Criterion._weight
    spec = {'weight': 0.1, 'ramp': [5, 10]}
    assert [round(w(spec, ep), 3) for ep in (1, 5, 6, 8, 10, 50)] == [0, 0, 0.02, 0.06, 0.1, 0.1]
