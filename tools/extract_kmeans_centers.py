"""KMeans centres of one feature node, per class, to initialise non-learnable prototypes.

  python tools/extract_kmeans_centers.py configs/pvt_linear.yaml checkpoints/pvt_linear.pth \
      --node dd4 --num_prototype 5 --out assets/kmeans/dd4_ch512_m5.pth
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protoseg_polyp import DEVICE  # noqa: E402
from protoseg_polyp.config import load_config  # noqa: E402
from protoseg_polyp.data import get_loader  # noqa: E402
from protoseg_polyp.models import build_model, load_checkpoint  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('checkpoint')
    ap.add_argument('--node', default='dd4', help='x1..x4, dd4..dd1, d1')
    ap.add_argument('--num_prototype', type=int, default=5)
    ap.add_argument('--max_per_batch', type=int, default=20000)
    ap.add_argument('--out', required=True)
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='config overrides')
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    model = build_model(cfg['model']).to(DEVICE).eval()
    load_checkpoint(model, args.checkpoint)
    loader = get_loader(cfg['data']['train_root'], 16, cfg['train']['img_size'])

    feats = {0: [], 1: []}
    for images, gts, _ in tqdm(loader):
        f = model.encoder(images.to(DEVICE))[args.node]
        b, c, h, w = f.shape
        f = F.normalize(F.layer_norm(f.permute(0, 2, 3, 1).reshape(-1, c), (c,)), p=2, dim=-1)
        lab = F.interpolate(gts.to(DEVICE), size=(h, w), mode='nearest').view(-1).long()
        for k in feats:
            fk = f[lab == k]
            if len(fk) > args.max_per_batch:
                fk = fk[torch.randperm(len(fk), device=fk.device)[:args.max_per_batch]]
            feats[k].append(fk.cpu().numpy())

    centers = torch.zeros(2, args.num_prototype, c)
    for k in feats:
        km = MiniBatchKMeans(n_clusters=args.num_prototype, batch_size=50000, n_init=3, max_iter=100)
        km.fit(np.concatenate(feats[k]))
        centers[k] = F.normalize(torch.from_numpy(km.cluster_centers_).float(), p=2, dim=-1)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    torch.save(centers, args.out)
    print(f'saved {tuple(centers.shape)} -> {args.out}')


if __name__ == '__main__':
    main()
