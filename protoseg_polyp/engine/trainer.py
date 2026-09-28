"""Training loop shared by every method (mirrors the original EMCAD polyp training script)."""
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from .. import DEVICE
from ..data import get_loader, list_pairs, split_train_val
from .evaluate import DATASETS, evaluate, evaluate_pairs
from .losses import Criterion
from .trackers import PrototypeTracker


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                p.grad.data.clamp_(-grad_clip, grad_clip)


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(model, cfg, out_dir, device=DEVICE):
    """cfg: the full experiment config (see configs/). Writes check.pth, best.pth, epoch_N.pth, train.log."""
    t, d = cfg['train'], cfg['data']
    os.makedirs(out_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(out_dir, 'train.log'), level=logging.INFO, filemode='a',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info('config: ' + json.dumps(cfg))

    if t.get('seed') is not None:
        seed_everything(t['seed'], t.get('deterministic', False))

    # Model selection: on a held-out part of the training set when data.val_fraction > 0; otherwise on the
    # test sets, which reproduces the original research code (and makes the reported numbers optimistic).
    val_fraction = d.get('val_fraction', 0.0)
    train_idx, val_pairs = None, None
    if val_fraction > 0:
        train_idx, val_idx = split_train_val(d['train_root'], val_fraction, seed=t.get('seed') or 0)
        images, gts = list_pairs(d['train_root'])
        val_pairs = ([images[i] for i in val_idx], [gts[i] for i in val_idx])
        logging.info(f'train/val split: {len(train_idx)} / {len(val_idx)} images')
    else:
        logging.info('no validation split: best.pth is selected on the test sets (original protocol)')

    loader = get_loader(d['train_root'], t['batchsize'], t['img_size'], t.get('augmentation', False),
                        d.get('superpixel_root'), indices=train_idx)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=t['lr'], weight_decay=t.get('weight_decay', 1e-4))
    criterion = Criterion(cfg['loss'], model.head)
    tracker = PrototypeTracker(model.head) if hasattr(model.head, 'fg_channels') else None
    rates = t.get('multiscale', [1])
    select = cfg.get('eval', {}).get('select_protocol', 'unified')
    best = -1.0

    for epoch in range(1, t['epochs'] + 1):
        model.train()
        warm = t.get('hard_warmup_epochs')
        alpha = min(1.0, epoch / warm) if warm else 1.0
        for step, (images, gts, sp) in enumerate(loader, start=1):
            for rate in rates:
                optimizer.zero_grad()
                x, y, s = images.to(device), gts.to(device), sp.to(device)
                if rate != 1:
                    size = int(round(t['img_size'] * rate / 32) * 32)
                    x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=True)
                    y = F.interpolate(y, size=(size, size), mode='bilinear', align_corners=True)
                out = model(x, gt=y.squeeze(1).long(), sp=s, alpha=alpha)
                loss, terms = criterion(out, y, epoch)
                loss.backward()
                clip_gradient(optimizer, t.get('clip', 0.5))
                optimizer.step()
            if tracker is not None and 'sim' in out:
                tracker.update(out['sim'].detach(), y, epoch)
            if step % 20 == 0 or step == len(loader):
                msg = f'Epoch [{epoch:03d}/{t["epochs"]:03d}], Step [{step:04d}/{len(loader):04d}], ' + \
                      ', '.join(f'{k}: {v:.4f}' for k, v in terms.items())
                print(msg, flush=True)
                logging.info(msg)

        torch.save(model.state_dict(), os.path.join(out_dir, 'check.pth'))
        if epoch in t.get('save_epochs', []):
            torch.save(model.state_dict(), os.path.join(out_dir, f'epoch_{epoch}.pth'))

        res = evaluate(model, d['test_root'], protocol=select, img_size=t['img_size'], device=device)
        for ds in DATASETS:
            logging.info(f'epoch: {epoch}, dataset: {ds}, dice: {res[ds]["dice"]}')
        logging.info(f'mDice: {res["mDice"]}')
        score, name = res['mDice'], 'test mDice'
        if val_pairs is not None:
            score, name = evaluate_pairs(model, val_pairs, select, t['img_size'], device)['dice'], 'val Dice'
            logging.info(f'epoch: {epoch}, val dice: {score}')
        print(f'epoch {epoch}: test mDice {res["mDice"]:.4f}' +
              (f' | val Dice {score:.4f}' if val_pairs is not None else ''), flush=True)
        if score > best:
            best = score
            torch.save(model.state_dict(), os.path.join(out_dir, 'best.pth'))
            logging.info(f'best {name} improved to {best} (epoch {epoch})')

    if tracker is not None:
        tracker.plot(out_dir)
    return best
