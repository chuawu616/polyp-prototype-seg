"""Shared fixtures. Every test runs on CPU without data or pretrained weights."""
import glob
import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

from protoseg_polyp.config import load_config  # noqa: E402

PVT_CONFIGS = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, 'configs', '[!_]*.yaml'))
                     if not os.path.basename(p).startswith('dinov3'))
DINO_CONFIGS = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, 'configs', 'dinov3*.yaml')))
RUN_DINOV3 = os.environ.get('PROTOSEG_TEST_DINOV3') == '1'  # needs network access to torch.hub


def offline_config(name):
    """Config with all pretrained weights / KMeans centres disabled."""
    cfg = load_config(os.path.join(ROOT, 'configs', name))
    enc = cfg['model']['encoder']
    enc['pretrained' if enc['type'] == 'pvt_emcad' else 'weights'] = None
    cfg['model']['head'].pop('kmeans_centers', None)
    return cfg


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


def _make_split(root, n, size=80, seed=0):
    rng = np.random.RandomState(seed)
    os.makedirs(os.path.join(root, 'images'))
    os.makedirs(os.path.join(root, 'masks'))
    for i in range(n):
        Image.fromarray(rng.randint(0, 255, (size, size, 3), dtype=np.uint8)).save(os.path.join(root, 'images', f'{i}.png'))
        m = np.zeros((size, size), np.uint8)
        m[20:60, 15:55] = 255
        Image.fromarray(m).save(os.path.join(root, 'masks', f'{i}.png'))


@pytest.fixture
def tiny_data(tmp_path):
    """PraNet-style folders with random images: TrainDataset (8) and five 2-image test sets."""
    from protoseg_polyp.engine.evaluate import DATASETS
    _make_split(str(tmp_path / 'TrainDataset'), 8)
    for i, ds in enumerate(DATASETS):
        _make_split(str(tmp_path / 'TestDataset' / ds), 2, seed=i + 1)
    return tmp_path
