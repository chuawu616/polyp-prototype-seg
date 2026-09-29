"""Where do the foreground sub-classes sit? Quantifies the layout of the FG sub-class map on the five test sets.

For every test image the FG sub-class of each GT-polyp pixel (argmax of the prototype similarity over the FG
channels, at 352x352) is compared with properties that do *not* need sub-class labels:
  * image level - share of the polyp taken by the dominant sub-class; which sub-class dominates vs. polyp size,
                  brightness and source dataset (Kruskal-Wallis, normalised mutual information)
  * depth       - distance to the GT boundary / max distance inside that polyp (0 = rim, 1 = centre);
                  concentric rings show up as sub-classes with ordered depth
  * position    - relative height / width inside the polyp bounding box (large polyps only);
                  horizontal bands show up as sub-classes with ordered height and equal width

  python tools/analyze_subclass_layout.py \
      --run "V0, original PPD"  configs/nonlearnable_v0_fb33.yaml     runs/followup/v0_ppd_original/best.pth \
      --run "V0, corrected PPD" configs/nonlearnable_v0_fb33.yaml     runs/followup/v0_ppd_corrected/best.pth \
      --run "Pseudo, SP 50 px"  configs/pseudo_fb33_superpixel.yaml runs/followup/pseudo_sp50/best.pth \
      --run "Pseudo, SP 800 px" configs/pseudo_fb33_superpixel.yaml runs/followup/pseudo_sp800/best.pth \
      --out docs/results/subclass_layout.md --figure docs/figures/followup_subclass_maps.jpg

The figure shows the sub-class maps of every run on --images (default: a small, a medium and a large polyp).
"""
import argparse
import os
import sys
from collections import defaultdict

import matplotlib
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.stats import kruskal
from sklearn.metrics import normalized_mutual_info_score

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_features import subclass_rgb  # noqa: E402
from protoseg_polyp import DEVICE  # noqa: E402
from protoseg_polyp.config import load_config  # noqa: E402
from protoseg_polyp.data import MEAN, STD, TestDataset  # noqa: E402
from protoseg_polyp.engine.evaluate import DATASETS  # noqa: E402
from protoseg_polyp.models import build_model, load_checkpoint  # noqa: E402

SIZE = 352
LARGE = 0.15  # a polyp covering more than 15 % of the image counts as large
IMAGES = ['CVC-ClinicDB/179.png', 'Kvasir/cju2wve9v7esz0878mxsdcy04.png', 'Kvasir/cju2nnqrqzp580855z8mhzgd6.png']


def layout(sub, mask, num_fg):
    """sub: (H, W) FG sub-class index; mask: (H, W) bool GT polyp.
    Returns pixel counts per sub-class and, per sub-class, depth / relative height / relative width of its pixels."""
    dist = distance_transform_edt(mask)
    depth = dist / dist.max()
    yy, xx = np.nonzero(mask)
    ry = (yy - yy.min()) / max(yy.max() - yy.min(), 1)
    rx = (xx - xx.min()) / max(xx.max() - xx.min(), 1)
    s = sub[yy, xx]
    return dict(counts=np.bincount(s, minlength=num_fg),
                depth=[depth[yy, xx][s == k] for k in range(num_fg)],
                y=[ry[s == k] for k in range(num_fg)], x=[rx[s == k] for k in range(num_fg)])


@torch.no_grad()
def collect(model, test_root):
    """One pass over the test sets: per-image records and pooled per-sub-class pixel statistics."""
    model.eval()
    fg = list(model.head.fg_channels)
    images = []
    pooled = {k: defaultdict(list) for k in ('small', 'large')}
    for ds in DATASETS:
        for image, gt, name in TestDataset(os.path.join(test_root, ds), SIZE):
            mask = np.asarray(gt.resize((SIZE, SIZE)), np.float32) / 255.0 > 0.5
            if not mask.any():
                continue
            sub = model(image.to(DEVICE))['sim'][0, fg].argmax(0).cpu().numpy()
            lay = layout(sub, mask, len(fg))
            rgb = (image[0].permute(1, 2, 0).numpy() * STD + MEAN).clip(0, 1)[mask]
            share = lay['counts'] / lay['counts'].sum()
            images.append(dict(ds=ds, name=name, area=mask.mean(), bright=rgb.mean(),
                               dom=int(share.argmax()), share=share.max()))
            size = 'large' if mask.mean() > LARGE else 'small'
            for key in ('depth', 'y', 'x'):
                pooled[size][key].append(lay[key])
    return images, pooled, len(fg)


def _median(chunks, k):
    v = np.concatenate([c[k] for c in chunks])
    return (np.median(v), len(v)) if len(v) else (float('nan'), 0)


def summarise(label, images, pooled, num_fg):
    dom = np.array([r['dom'] for r in images])
    share = np.array([r['share'] for r in images])
    area = np.array([r['area'] for r in images])
    bright = np.array([r['bright'] for r in images])

    def kw(v):
        groups = [v[dom == k] for k in range(num_fg) if (dom == k).sum() > 5]
        return f'{kruskal(*groups).pvalue:.1e}' if len(groups) > 1 else '–'

    lines = [f'### {label}', '',
             f'{len(images)} test images; the dominant FG sub-class takes on average {share.mean():.2f} of the polyp; '
             f'> 90 % in {(share > 0.9).mean():.0%} of images, > 70 % in {(share > 0.7).mean():.0%}. '
             f'Dominant sub-class vs. area: Kruskal-Wallis p = {kw(area)}; vs. brightness: p = {kw(bright)}; '
             f'NMI with the source dataset = '
             f'{normalized_mutual_info_score([r["ds"] for r in images], dom):.3f}.', '',
             '| FG sub-class | images dominated | median polyp area | median brightness '
             '| depth, small polyps (pixel share) | depth, large polyps (pixel share) '
             '| height in large polyps | width in large polyps |',
             '|---|---|---|---|---|---|---|---|']
    tot = {s: sum(sum(len(d) for d in c) for c in pooled[s]['depth']) or 1 for s in pooled}
    for k in range(num_fg):
        sel = dom == k
        ds_, ns = _median(pooled['small']['depth'], k)
        dl, nl = _median(pooled['large']['depth'], k)
        y, _ = _median(pooled['large']['y'], k)
        x, _ = _median(pooled['large']['x'], k)
        lines.append(f'| FG{k} | {sel.sum()} | {np.median(area[sel]) if sel.any() else float("nan"):.3f} '
                     f'| {np.median(bright[sel]) if sel.any() else float("nan"):.3f} '
                     f'| {ds_:.2f} ({ns / tot["small"]:.2f}) | {dl:.2f} ({nl / tot["large"]:.2f}) | {y:.2f} | {x:.2f} |')
    n_large = sum(r['area'] > LARGE for r in images)
    lines += ['', f'Depth: 0 = boundary, 1 = centre. Height / width: 0 = top / left of the polyp bounding box; '
                  f'{n_large} large polyps (> {LARGE:.0%} of the image).', '']
    return '\n'.join(lines)


@torch.no_grad()
def figure(runs, test_root, names, out):
    fig, axes = plt.subplots(len(names), 2 + len(runs), figsize=(2.3 * (2 + len(runs)), 2.4 * len(names)),
                             squeeze=False)
    for c, (label, model) in enumerate(runs):
        model.eval()
        for r, rel in enumerate(names):
            ds, name = rel.split('/')
            data = TestDataset(os.path.join(test_root, ds), SIZE)
            i = [os.path.basename(p) for p in data.images].index(name)
            data.images, data.gts = data.images[i:i + 1], data.gts[i:i + 1]
            image, gt, _ = next(iter(data))
            gt = np.asarray(gt.resize((SIZE, SIZE)))
            if c == 0:
                axes[r, 0].imshow((image[0].permute(1, 2, 0).numpy() * STD + MEAN).clip(0, 1))
                axes[r, 0].set_ylabel(f'{ds}\npolyp {(gt > 127).mean():.0%} of image', fontsize=8)
                axes[r, 1].imshow(gt, cmap='gray')
            axes[r, 2 + c].imshow(subclass_rgb(model(image.to(DEVICE))['sim'][0], model.head))
            axes[0, 2 + c].set_title(label, fontsize=9)
    axes[0, 0].set_title('image', fontsize=9)
    axes[0, 1].set_title('GT', fontsize=9)
    for a in axes.ravel():
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    plt.savefig(out, dpi=120)
    print('saved', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', nargs=3, action='append', required=True, metavar=('LABEL', 'CONFIG', 'CHECKPOINT'))
    ap.add_argument('--out', default=None, help='markdown report')
    ap.add_argument('--figure', default=None)
    ap.add_argument('--images', nargs='*', default=IMAGES, help='<dataset>/<file> for the figure rows')
    args = ap.parse_args()
    runs, parts = [], ['# Layout of the foreground sub-classes (test sets, prototype similarity argmax)', '']
    for label, cfg_path, ckpt in args.run:
        cfg = load_config(cfg_path)
        model = build_model(cfg['model']).to(DEVICE)
        load_checkpoint(model, ckpt)
        text = summarise(f'{label} (`{ckpt}`)', *collect(model, cfg['data']['test_root']))
        print(text, flush=True)
        parts.append(text)
        runs.append((label, model))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            f.write('\n'.join(parts))
    if args.figure:
        figure(runs, load_config(args.run[0][1])['data']['test_root'], args.images, args.figure)


if __name__ == '__main__':
    main()
