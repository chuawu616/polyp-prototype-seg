import numpy as np
from scipy.ndimage import distance_transform_edt

from analyze_subclass_layout import layout


def _disc(n=64, r=24):
    yy, xx = np.mgrid[:n, :n]
    return np.hypot(yy - n / 2, xx - n / 2) < r, yy


def test_rings_are_ordered_by_depth():
    mask, _ = _disc()
    dist = distance_transform_edt(mask)
    sub = np.digitize(dist / dist.max(), [1 / 3, 2 / 3])    # rim -> 0, middle -> 1, centre -> 2
    lay = layout(sub, mask, 3)
    med = [np.median(v) for v in lay['depth']]
    assert med[0] < med[1] < med[2] and lay['counts'].sum() == mask.sum()
    assert abs(np.median(lay['y'][2]) - 0.5) < 0.05


def test_bands_are_ordered_by_height_not_width():
    mask, yy = _disc()
    sub = np.digitize(yy, [64 / 2 - 8, 64 / 2 + 8])          # top, middle, bottom bands
    lay = layout(sub, mask, 3)
    ys = [np.median(v) for v in lay['y']]
    xs = [np.median(v) for v in lay['x']]
    assert ys[0] < ys[1] < ys[2] and max(abs(x - 0.5) for x in xs) < 0.05
