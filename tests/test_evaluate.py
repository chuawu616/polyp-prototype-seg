import numpy as np
import pytest

from protoseg_polyp.engine.evaluate import image_scores

GT = np.zeros((20, 20), np.float32)
GT[5:15, 5:15] = 1


@pytest.mark.parametrize('protocol', ['unified', 'legacy_fg_only', 'legacy_softmax'])
def test_hard_dice(protocol):
    assert image_scores(GT.copy(), GT, protocol)[0] == pytest.approx(1.0)
    half = GT.copy()
    half[5:15, 10:15] = 0
    dice, iou = image_scores(half, GT, protocol)
    assert dice == pytest.approx(2 * 50 / 150) and iou == pytest.approx(0.5)


def test_unified_thresholds_without_minmax_but_legacy_normalises():
    low = GT * 0.4                          # every probability < 0.5
    assert image_scores(low, GT, 'unified')[0] == pytest.approx(0.0)
    assert image_scores(low, GT, 'legacy_softmax')[0] == pytest.approx(1.0)   # min-max rescales to [0, 1]


def test_legacy_soft_dice_matches_original_formula():
    pred = GT * 0.8 + 0.1
    p = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    expected = float('{:.4f}'.format((2 * (p * GT).sum() + 1) / (p.sum() + GT.sum() + 1)))
    assert image_scores(pred, GT, 'legacy_soft')[0] == expected
