"""Prototype-learning primitives shared by the non-learnable heads.

`distributed_sinkhorn` and `momentum_update` follow ProtoSeg
(Zhou et al., "Rethinking Semantic Segmentation: A Prototype View", CVPR 2022).
"""
import torch
import torch.nn.functional as F


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)


def momentum_update(old_value, new_value, momentum):
    """EMA: new = m * old + (1 - m) * new."""
    return momentum * old_value + (1 - momentum) * new_value


def distributed_sinkhorn(out, sinkhorn_iterations=3, epsilon=0.05):
    """Balanced assignment of N samples to M prototypes (Sinkhorn-Knopp).

    Args:
        out: (N, M) similarity between samples and the M sub-prototypes of one class.
    Returns:
        q:      (N, M) hard one-hot assignment, *sampled* with Gumbel-softmax (used for the EMA update)
        indexs: (N,)   argmax of the balanced transport plan (used as the sub-class label)

    Note: q and indexs can disagree because q is a stochastic sample - this mirrors ProtoSeg.
    """
    L = torch.exp(out / epsilon).t()  # (M, N)
    B = L.shape[1]
    K = L.shape[0]

    L /= torch.sum(L)
    for _ in range(sinkhorn_iterations):
        L /= torch.sum(L, dim=1, keepdim=True)  # each prototype gets total mass 1/K
        L /= K
        L /= torch.sum(L, dim=0, keepdim=True)  # each sample gets total mass 1/B
        L /= B
    L *= B
    L = L.t()

    indexs = torch.argmax(L, dim=1)
    L = F.gumbel_softmax(L, tau=0.5, hard=True)
    return L, indexs
