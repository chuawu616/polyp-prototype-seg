"""Prototype-utilisation diagnostics recorded during training (the curves used to diagnose collapse).

Every pixel is assigned to its most similar prototype (argmax over all prototypes, as in the original
scripts). Per class we record
  * ratio curve        - share of pixels won by each prototype, per epoch
  * image-level collapse - share of images in which one prototype wins > 50 / 70 / 90 % of the pixels
  * step competition   - per-step share of each prototype (moving average), shows collapse at step 1
"""
import os
from collections import defaultdict

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

THRESHOLDS = (0.5, 0.7, 0.9)


class PrototypeTracker:
    def __init__(self, head):
        self.groups = {'FG': list(head.fg_channels), 'BG': list(head.bg_channels)}
        self.epoch_counts = defaultdict(lambda: {g: np.zeros(len(c)) for g, c in self.groups.items()})
        self.collapse = defaultdict(lambda: {g: np.zeros(len(THRESHOLDS) + 1) for g in self.groups})
        self.steps = {g: [] for g in self.groups}

    @torch.no_grad()
    def update(self, sim, gt, epoch):
        win = sim.argmax(dim=1).view(sim.shape[0], -1).cpu().numpy()   # (B, HW) prototype index
        for g, chans in self.groups.items():
            step = np.zeros(len(chans))
            for img in win:
                counts = np.array([(img == c).sum() for c in chans], dtype=float)
                step += counts
                self.epoch_counts[epoch][g] += counts
                if counts.sum() > 0:
                    r = counts.max() / counts.sum()
                    self.collapse[epoch][g][:-1] += [r > t for t in THRESHOLDS]
                    self.collapse[epoch][g][-1] += 1
            self.steps[g].append(step / max(step.sum(), 1))

    def plot(self, out_dir):
        epochs = sorted(self.epoch_counts)
        fig, axes = plt.subplots(3, 2, figsize=(16, 15))
        for col, g in enumerate(self.groups):
            ratios = np.array([self.epoch_counts[e][g] / max(self.epoch_counts[e][g].sum(), 1) for e in epochs])
            for k in range(ratios.shape[1]):
                axes[0, col].plot(epochs, ratios[:, k], marker='.', label=f'{g}_{k}')
            axes[0, col].set_title(f'{g} prototype pixel ratio')
            coll = np.array([self.collapse[e][g][:-1] / max(self.collapse[e][g][-1], 1) for e in epochs])
            for i, t in enumerate(THRESHOLDS):
                axes[1, col].plot(epochs, coll[:, i], marker='o', label=f'>{int(t * 100)}% one prototype')
            axes[1, col].set_title(f'{g} image-level collapse')
            steps = np.array(self.steps[g])
            n = min(50, len(steps))
            smooth = np.array([np.convolve(steps[:, k], np.ones(n) / n, mode='valid') for k in range(steps.shape[1])])
            for k in range(smooth.shape[0]):
                axes[2, col].plot(smooth[k], label=f'{g}_{k}')
            axes[2, col].set_title(f'{g} step-wise competition (moving avg {n})')
            for r in range(3):
                axes[r, col].set_ylim(-0.05, 1.05)
                axes[r, col].grid(True, linestyle='--', alpha=0.5)
                axes[r, col].legend(fontsize=8)
        axes[2, 0].set_xlabel('step')
        axes[1, 0].set_xlabel('epoch')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'prototype_diagnostics.png'), dpi=120)
        plt.close()
