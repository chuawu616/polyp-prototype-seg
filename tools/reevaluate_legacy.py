"""Re-evaluate every checkpoint left by the original research code (one-off, run on the lab server).

For each best.pth:
  1. infer the architecture from its state_dict (keys + shapes) and the folder name,
  2. load it into the refactored model with strict=True,
  3. evaluate with the protocol the original script used AND with the unified protocol (one forward pass),
  4. search every epoch of every training log for the per-dataset scores reproduced in (3),
     which both verifies the refactor and recovers which log/epoch the checkpoint came from.

Usage (GPU inference only):
  CUDA_VISIBLE_DEVICES=0 python tools/reevaluate_legacy.py --root /path/to/wch_code \
      --test_root /path/to/TestDataset --out docs/results/reevaluation.csv
"""
import argparse
import csv
import glob
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_logs import parse  # noqa: E402
from protoseg_polyp.engine.evaluate import DATASETS, evaluate  # noqa: E402
from protoseg_polyp.models import build_model, load_checkpoint  # noqa: E402

CONCAT = [1, 3, 5, 7, 9, 11]


def split_fb(name, total):
    """'fb_88' -> (8, 8), 'fb_1010' -> (10, 10), 'fb_84' -> (8, 4)."""
    m = re.search(r'fb_?(\d+)', name)
    d = m.group(1)
    fg, bg = (int(d[0]), int(d[1])) if len(d) == 2 else (int(d[:len(d) // 2]), int(d[len(d) // 2:]))
    assert fg + bg == total, (name, total)
    return fg, bg


def infer(sd, path):
    """Return (model config, legacy protocol) for an original-code checkpoint."""
    name = os.path.basename(os.path.dirname(path))
    if any(k.startswith('backbone.blocks.') for k in sd):  # DINOv3
        dim = sd['backbone.patch_embed.proj.weight'].shape[0]
        arch = next(a for a in ('vits16plus', 'vitb16', 'vitl16', 'vits16') if f'/{a}/' in path)
        first = next(sd[k] for k in ('head.0.weight', 'proj.0.weight', 'proto_proj.0.weight') if k in sd)
        encoder = dict(type='dinov3', arch=arch, layers=1 if first.shape[1] == dim else CONCAT)
        neck = 256 if any(k.startswith(('proj.', 'proto_proj.')) for k in sd) else None
    else:
        encoder, neck = dict(type='pvt_emcad'), None

    if any(k in sd for k in ('out_head4.weight', 'temp_head.weight', 'head.weight')):
        head = dict(type='linear')
    elif 'head.0.weight' in sd:
        head = dict(type='mlp')
    elif 'easy_prototypes' in sd:
        e, b = sd['easy_prototypes'].shape[0], sd['bg_prototypes'].shape[0]
        hybrid = 'hard_prototypes_module.global_hard' in sd
        h = sd['hard_prototypes_module.global_hard' if hybrid else 'hard_prototypes'].shape[0]
        head = dict(type='specialize', num_fg=e + h, num_hard=h, num_bg=b, hybrid=hybrid)
    elif sd['prototypes'].dim() == 2:
        fg, bg = split_fb(name, sd['prototypes'].shape[0])
        head = dict(type='prototype', num_fg=fg, num_bg=bg)
    else:
        M = sd['prototypes'].shape[1]
        if 'seg_head.weight' in sd:
            head = dict(type='pseudo', num_prototype=M)
        elif 'proto_proj_mlp.0.weight' in sd:
            head = dict(type='nonlearnable_v3', num_prototype=M, node='x4' if name.endswith('x4') else 'dd4')
        elif 'mask_norm.weight' in sd:
            head = dict(type='nonlearnable_v2', num_prototype=M)
        else:
            head = dict(type='nonlearnable_v0', num_prototype=M)

    protocol = {'pseudo': 'legacy_softmax', 'nonlearnable_v3': 'legacy_fg_only',
                'nonlearnable_v0': 'legacy_fg_only'}.get(head['type'], 'legacy_soft')
    if encoder['type'] == 'pvt_emcad':
        encoder['detach_taps'] = head['type'] != 'nonlearnable_v3'
    return dict(encoder=encoder, neck=neck, head=head), protocol


def log_epochs(root):
    """All (log, run, epoch, {dataset: dice}) from every training log."""
    out = []
    for log in glob.glob(os.path.join(root, '*', '**', '*.log'), recursive=True):
        if '.ipynb_checkpoints' in log or 'ProtoSeg' in log:
            continue
        for r, run in enumerate(parse(log)):
            for ep, d in run['epochs'].items():
                if all(ds in d for ds in DATASETS):
                    out.append((os.path.relpath(log, root), r + 1, ep, d))
    return out


def match(res, epochs):
    """Closest (log, run, epoch, max |dice difference|) over every logged epoch."""
    best = None
    for log, run, ep, d in epochs:
        diff = max(abs(res[ds]['dice'] - d[ds]) for ds in DATASETS)
        if best is None or diff < best[3]:
            best = (log, run, ep, diff)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='folder containing cascade/, dino_v3/, non_learnable/')
    ap.add_argument('--test_root', required=True)
    ap.add_argument('--out', default='docs/results/reevaluation.csv')
    ap.add_argument('--filter', default='', help='only checkpoints whose path contains this string')
    args = ap.parse_args()

    ckpts = sorted(glob.glob(os.path.join(args.root, '*', 'models', '**', 'best.pth'), recursive=True))
    done = set()
    if os.path.exists(args.out):
        done = {r['checkpoint'] for r in csv.DictReader(open(args.out))}
    epochs = log_epochs(args.root)
    fields = ['checkpoint', 'encoder', 'head', 'legacy_protocol', 'legacy_mDice', 'log', 'log_run', 'log_epoch',
              'log_max_abs_diff', 'verified'] + [f'{d}' for d in DATASETS] + ['mDice', 'mIoU', 'in_domain', 'out_domain']
    new_file = not os.path.exists(args.out)
    with open(args.out, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        for path in ckpts:
            rel = os.path.relpath(os.path.dirname(path), args.root)
            if rel in done or args.filter not in rel:
                continue
            t = time.time()
            sd = torch.load(path, map_location='cpu')
            cfg, protocol = infer(sd, path)
            model = build_model(cfg).cuda()
            try:
                load_checkpoint(model, path, legacy=True, strict=True)
            except RuntimeError as e:
                print(f'{rel}: SKIPPED, state_dict does not match the inferred architecture '
                      f'({str(e).splitlines()[0]})', flush=True)
                continue
            res = evaluate(model, args.test_root, protocol=[protocol, 'unified'])
            leg, uni = res[protocol], res['unified']
            m = match(leg, epochs)
            row = dict(checkpoint=rel, encoder=cfg['encoder'].get('arch', 'pvt_emcad') +
                       ('_concat' if cfg['encoder'].get('layers') == CONCAT else ''),
                       head=' '.join(f'{k}={v}' for k, v in cfg['head'].items()),
                       legacy_protocol=protocol, legacy_mDice=round(leg['mDice'], 4),
                       log=m[0], log_run=m[1], log_epoch=m[2], log_max_abs_diff=f'{m[3]:.5f}',
                       verified=m[3] < 2e-4, mDice=round(uni['mDice'], 4), mIoU=round(uni['mIoU'], 4),
                       in_domain=round(uni['in_domain'], 4), out_domain=round(uni['out_domain'], 4),
                       **{d: round(uni[d]['dice'], 4) for d in DATASETS})
            w.writerow(row)
            f.flush()
            print(f"{rel:90s} legacy {leg['mDice']:.4f} unified {uni['mDice']:.4f} "
                  f"match={row['verified']} ({m[3]:.5f} {m[0]} ep{m[2]}) {time.time() - t:.0f}s", flush=True)
            del model
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
