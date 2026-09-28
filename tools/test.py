"""Evaluate a checkpoint on the five test sets and optionally save masks / sub-class maps.

  python tools/test.py configs/pvt_proto_fb88.yaml checkpoints/pvt_proto_fb88.pth \
      [--protocol unified|legacy_soft|legacy_fg_only|legacy_softmax] [--save_dir preds/]

Checkpoints produced by the original research code are detected and converted automatically.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protoseg_polyp import DEVICE  # noqa: E402
from protoseg_polyp.config import load_config  # noqa: E402
from protoseg_polyp.data import TestDataset  # noqa: E402
from protoseg_polyp.engine.evaluate import DATASETS, evaluate  # noqa: E402
from protoseg_polyp.models import build_model, load_checkpoint  # noqa: E402


@torch.no_grad()
def save_predictions(model, test_root, save_dir, img_size):
    """<save_dir>/<dataset>/mask/<name>.png (binary) and .../subclass/<name>.png (argmax prototype index)."""
    model.eval()
    for ds in DATASETS:
        for image, gt, name in TestDataset(os.path.join(test_root, ds), img_size):
            out = model(image.to(DEVICE))
            size = gt.size[::-1]
            prob = F.interpolate(out['fg_prob'], size=size, mode='bilinear', align_corners=False)[0, 0]
            os.makedirs(os.path.join(save_dir, ds, 'mask'), exist_ok=True)
            Image.fromarray(((prob >= 0.5).cpu().numpy() * 255).astype(np.uint8)).save(
                os.path.join(save_dir, ds, 'mask', name))
            if 'sim' in out:
                sim = F.interpolate(out['sim'], size=size, mode='bilinear', align_corners=False)[0]
                os.makedirs(os.path.join(save_dir, ds, 'subclass'), exist_ok=True)
                Image.fromarray(sim.argmax(0).cpu().numpy().astype(np.uint8)).save(
                    os.path.join(save_dir, ds, 'subclass', name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('checkpoint')
    ap.add_argument('--protocol', default='unified')
    ap.add_argument('--save_dir', default=None)
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='config overrides')
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    model = build_model(cfg['model']).to(DEVICE)
    print(load_checkpoint(model, args.checkpoint))
    size = cfg['train']['img_size']
    res = evaluate(model, cfg['data']['test_root'], protocol=args.protocol, img_size=size)
    for ds in DATASETS:
        print(f'{ds:18s} dice {res[ds]["dice"]:.4f}  iou {res[ds]["iou"]:.4f}  (n={res[ds]["n"]})')
    print(f'mDice {res["mDice"]:.4f} | in-domain {res.get("in_domain", float("nan")):.4f} | '
          f'out-of-domain {res.get("out_domain", float("nan")):.4f}')
    if args.save_dir:
        save_predictions(model, cfg['data']['test_root'], args.save_dir, size)
        with open(os.path.join(args.save_dir, 'metrics.json'), 'w') as f:
            json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
