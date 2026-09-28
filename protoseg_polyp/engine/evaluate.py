"""Evaluation on the five polyp test sets.

Protocols
---------
unified      (default, used for every table in docs/)  p = foreground probability, bilinearly resized to
             the GT size, binarised at 0.5; per-image Dice and IoU. The same decision rule for every model.

Legacy protocols reproduce what each original training script logged, so old checkpoints can be checked
number-for-number against their logs:
legacy_soft        EMCAD / learnable-prototype / DINOv3 scripts: sigmoid(fg_logit), per-image min-max
                   normalisation, *soft* Dice with smooth=1, rounded to 4 decimals per image.
legacy_fg_only     non-learnable V0/V3 scripts: sigmoid of the FG max-cosine only (BG ignored), min-max,
                   threshold 0.5, hard Dice.
legacy_softmax     pseudo-label scripts: FG softmax probability, min-max, threshold 0.5, hard Dice.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from .. import DEVICE
from ..data import TestDataset

DATASETS = ['CVC-ClinicDB', 'Kvasir', 'CVC-300', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
IN_DOMAIN, OUT_DOMAIN = DATASETS[:2], DATASETS[2:]


def _minmax(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-8)


def _map(out, protocol, size):
    """Model output -> (H, W) numpy map at GT resolution, as the protocol defines it."""
    def up(t):
        return F.interpolate(t, size=size, mode='bilinear', align_corners=False)

    if protocol == 'unified':
        t = up(out['fg_prob'])
    elif protocol == 'legacy_soft':
        t = torch.sigmoid(up(out['fg_logit']))
    elif protocol == 'legacy_fg_only':
        t = torch.sigmoid(up(out['scores'][:, 1:2]))
    elif protocol == 'legacy_softmax':
        t = up(out['binary'][:, 1:2])
    else:
        raise ValueError(protocol)
    return t.squeeze().float().cpu().numpy()


def image_scores(pred, gt, protocol):
    if protocol == 'legacy_soft':
        pred = _minmax(pred)
        dice = (2 * (pred * gt).sum() + 1) / (pred.sum() + gt.sum() + 1)
        return float('{:.4f}'.format(dice)), float('nan')
    if protocol != 'unified':
        pred = _minmax(pred)
    pred = (pred >= 0.5).astype(np.float64)
    inter = (pred * gt).sum()
    dice = 2 * inter / (pred.sum() + gt.sum() + 1e-8)
    iou = inter / (pred.sum() + gt.sum() - inter + 1e-8)
    return dice, iou


def _summarise(per_ds, datasets):
    res = {ds: dict(dice=float(np.mean(v['dice'])), iou=float(np.mean(v['iou'])), n=len(v['dice']))
           for ds, v in per_ds.items()}
    res['mDice'] = float(np.mean([res[d]['dice'] for d in datasets]))
    if set(datasets) == set(DATASETS):
        res['mIoU'] = float(np.mean([res[d]['iou'] for d in datasets]))
        res['in_domain'] = float(np.mean([res[d]['dice'] for d in IN_DOMAIN]))
        res['out_domain'] = float(np.mean([res[d]['dice'] for d in OUT_DOMAIN]))
    return res


def _score(model, loader, protocols, device):
    acc = {p: dict(dice=[], iou=[]) for p in protocols}
    for image, gt, name in loader:
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        out = model(image.to(device))
        for p in protocols:
            d, i = image_scores(_map(out, p, gt.shape), gt, p)
            acc[p]['dice'].append(d)
            acc[p]['iou'].append(i)
    return acc


@torch.no_grad()
def evaluate(model, test_root, protocol='unified', img_size=352, datasets=DATASETS, device=DEVICE):
    """Returns {dataset: {'dice', 'iou', 'n'}, 'mDice', 'mIoU', 'in_domain', 'out_domain'}.

    `protocol` may also be a list, in which case one forward pass per image serves every protocol
    and the result is {protocol: summary}.
    """
    protocols = [protocol] if isinstance(protocol, str) else list(protocol)
    model.eval()
    per_ds = {ds: _score(model, TestDataset(os.path.join(test_root, ds), img_size), protocols, device)
              for ds in datasets}
    res = {p: _summarise({ds: per_ds[ds][p] for ds in datasets}, datasets) for p in protocols}
    return res[protocol] if isinstance(protocol, str) else res


@torch.no_grad()
def evaluate_pairs(model, pairs, protocol='unified', img_size=352, device=DEVICE):
    """Mean Dice / IoU over explicit (images, masks) path lists, e.g. a validation split."""
    model.eval()
    acc = _score(model, TestDataset(None, img_size, pairs=pairs), [protocol], device)[protocol]
    return dict(dice=float(np.mean(acc['dice'])), iou=float(np.mean(acc['iou'])), n=len(acc['dice']))
