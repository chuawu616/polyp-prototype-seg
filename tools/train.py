"""Train one experiment.

  python tools/train.py configs/pvt_proto_fb88.yaml [--out runs/<name>] [--set KEY=VALUE ...]

Examples of overrides:  --set data.train_root=/data/polyp/TrainDataset train.epochs=10 data.val_fraction=0.1
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protoseg_polyp import DEVICE  # noqa: E402
from protoseg_polyp.config import load_config  # noqa: E402
from protoseg_polyp.engine.trainer import train  # noqa: E402
from protoseg_polyp.models import build_model, load_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--out', default=None)
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='config overrides')
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    out_dir = args.out or os.path.join('runs', os.path.splitext(os.path.basename(args.config))[0])

    model = build_model(cfg['model']).to(DEVICE)
    init = cfg['train'].get('init_from')
    if init:  # e.g. structure-loss-pretrained linear baseline: load every shape-compatible weight
        print(load_checkpoint(model, init, shape_match=True))
    best = train(model, cfg, out_dir)
    metric = 'val Dice' if cfg['data'].get('val_fraction', 0) > 0 else 'test mDice'
    print(f'best {metric} {best:.4f} -> {out_dir}/best.pth')


if __name__ == '__main__':
    main()
