"""Feature diagnostics: prediction, prototype sub-class map and a PCA->RGB view of every feature node.

Two modes, each row = image | GT | prediction | sub-class argmax | PCA(node) for each node:
  * worst / best k images of one checkpoint
      python tools/analyze_features.py configs/pseudo_fb33.yaml ckpt.pth --worst 10 --out figs/worst.jpg
  * evolution of one image across checkpoints (e.g. epoch_1.pth ... epoch_100.pth)
      python tools/analyze_features.py configs/pseudo_fb33.yaml runs/x/epoch_{1,5,20,100}.pth \
          --image data/polyp/TestDataset/Kvasir/images/xyz.png --out figs/evolution.jpg

Sub-class colours: background prototypes in dark tones, foreground prototypes in bright tones.
"""
import argparse
import os
import sys

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protoseg_polyp import DEVICE  # noqa: E402
from protoseg_polyp.config import load_config  # noqa: E402
from protoseg_polyp.data import MEAN, STD, TestDataset  # noqa: E402
from protoseg_polyp.engine.evaluate import DATASETS, image_scores  # noqa: E402
from protoseg_polyp.models import build_model, load_checkpoint  # noqa: E402

PVT_NODES = ['x1', 'x2', 'x3', 'x4', 'dd4', 'dd3', 'dd2', 'dd1', 'd1']


def pca_rgb(feat, size):
    c, h, w = feat.shape
    x = feat.permute(1, 2, 0).reshape(-1, c).cpu().numpy()
    y = PCA(n_components=3).fit_transform(x)
    y = (y - y.min(0)) / (y.max(0) - y.min(0) + 1e-8)
    img = torch.from_numpy(y.reshape(h, w, 3)).permute(2, 0, 1)[None].float()
    return F.interpolate(img, size=size, mode='bilinear', align_corners=False)[0].permute(1, 2, 0).numpy()


def subclass_rgb(sim, head):
    idx = sim.argmax(0).cpu().numpy()
    fg, bg = list(head.fg_channels), list(head.bg_channels)
    fg_cols = plt.cm.autumn(np.linspace(0, 1, len(fg)))[:, :3]
    bg_cols = plt.cm.winter(np.linspace(0, 1, len(bg)))[:, :3] * 0.5
    lut = np.zeros((max(fg + bg) + 1, 3))
    lut[fg], lut[bg] = fg_cols, bg_cols
    return lut[idx]


@torch.no_grad()
def row(model, image, gt, size):
    feats = model.encoder(image.to(DEVICE))
    out = model(image.to(DEVICE))
    prob = out['fg_prob'][0, 0].cpu().numpy()
    cells = [(np.clip(image[0].permute(1, 2, 0).numpy() * STD + MEAN, 0, 1), 'image'),
             (np.asarray(gt.resize(size[::-1])) / 255.0, 'GT'),
             (prob >= 0.5, f'pred'),
             (subclass_rgb(out['sim'][0], model.head) if 'sim' in out else np.zeros((*size, 3)), 'sub-class')]
    for n in [k for k in PVT_NODES if k in feats] or [model.encoder.out_key]:
        cells.append((pca_rgb(feats[n][0], size), f'PCA {n}'))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('checkpoints', nargs='+')
    ap.add_argument('--image', default=None, help='evolution mode: one image across checkpoints')
    ap.add_argument('--worst', type=int, default=0)
    ap.add_argument('--best', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='config overrides')
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    size = (cfg['train']['img_size'],) * 2
    model = build_model(cfg['model']).to(DEVICE).eval()

    rows, labels = [], []
    if args.image:
        tf = T.Compose([T.Resize(size), T.ToTensor(), T.Normalize(MEAN, STD)])
        image = tf(Image.open(args.image).convert('RGB')).unsqueeze(0)
        gt = Image.open(args.image.replace('/images/', '/masks/').rsplit('.', 1)[0] + '.png').convert('L')
        for ck in args.checkpoints:
            load_checkpoint(model, ck)
            rows.append(row(model, image, gt, size))
            labels.append(os.path.basename(ck))
    else:
        load_checkpoint(model, args.checkpoints[0])
        scored = []
        for ds in DATASETS:
            for image, gt, name in TestDataset(os.path.join(cfg['data']['test_root'], ds), size[0]):
                with torch.no_grad():
                    p = F.interpolate(model(image.to(DEVICE))['fg_prob'], size=gt.size[::-1], mode='bilinear')
                g = np.asarray(gt, np.float32) / 255.0
                scored.append((image_scores(p[0, 0].cpu().numpy(), (g > 0.5).astype(np.float32), 'unified')[0],
                               ds, name, image, gt))
        scored.sort(key=lambda s: s[0])
        picked = scored[:args.worst] + (scored[-args.best:] if args.best else [])
        for dice, ds, name, image, gt in picked:
            rows.append(row(model, image, gt, size))
            labels.append(f'{ds}/{name} Dice {dice:.3f}')

    ncol = len(rows[0])
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.2 * ncol, 2.3 * len(rows)), squeeze=False)
    for r, (cells, label) in enumerate(zip(rows, labels)):
        for c, (img, title) in enumerate(cells):
            axes[r, c].imshow(img, cmap='gray' if img.ndim == 2 else None)
            axes[r, c].set_title(title if r == 0 else '', fontsize=9)
            axes[r, c].axis('off')
        axes[r, 0].text(-10, size[0] / 2, label, rotation=90, va='center', ha='right', fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.savefig(args.out, dpi=110)
    print('saved', args.out)


if __name__ == '__main__':
    main()
