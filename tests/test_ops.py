import torch

from protoseg_polyp.ops import distributed_sinkhorn, l2_normalize, momentum_update


def test_sinkhorn_balances_a_skewed_similarity():
    # every sample prefers prototype 0; a balanced assignment must still use all prototypes
    n, m = 600, 3
    sim = torch.randn(n, m) * 0.01
    sim[:, 0] += 0.2
    q, idx = distributed_sinkhorn(sim)
    assert q.shape == (n, m) and idx.shape == (n,)
    counts = torch.bincount(idx, minlength=m).float()
    assert (counts > n / m * 0.5).all(), counts        # no prototype starves
    assert torch.allclose(q.sum(1), torch.ones(n))     # hard one-hot rows


def test_sinkhorn_keeps_a_clear_structure():
    # three well separated groups are recovered exactly
    sim = torch.full((300, 3), -1.0)
    for k in range(3):
        sim[k * 100:(k + 1) * 100, k] = 1.0
    _, idx = distributed_sinkhorn(sim)
    assert torch.equal(idx, torch.arange(3).repeat_interleave(100))


def test_momentum_update_and_normalize():
    old, new = torch.zeros(4), torch.ones(4)
    assert torch.allclose(momentum_update(old, new, 0.9), torch.full((4,), 0.1))
    assert torch.allclose(l2_normalize(torch.randn(5, 7)).norm(dim=-1), torch.ones(5))
