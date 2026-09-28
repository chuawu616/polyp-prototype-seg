import os

import torch

from conftest import ROOT
from parse_logs import parse
from protoseg_polyp.config import load_config
from protoseg_polyp.data import PolypDataset, TestDataset, split_train_val


def test_split_is_deterministic_and_disjoint(tiny_data):
    root = str(tiny_data / 'TrainDataset')
    tr, va = split_train_val(root, 0.25, seed=3)
    assert (tr, va) == split_train_val(root, 0.25, seed=3)
    assert len(va) == 2 and not set(tr) & set(va) and sorted(tr + va) == list(range(8))


def test_datasets(tiny_data):
    ds = PolypDataset(str(tiny_data / 'TrainDataset'), trainsize=64, augmentation=True, indices=[0, 1, 2])
    img, gt, sp = ds[0]
    assert len(ds) == 3 and img.shape == (3, 64, 64) and gt.shape == (1, 64, 64) and (sp == -1).all()
    image, mask, name = next(iter(TestDataset(str(tiny_data / 'TestDataset' / 'Kvasir'), 64)))
    assert image.shape == (1, 3, 64, 64) and mask.size == (80, 80) and name.endswith('.png')


def test_config_inheritance_and_overrides():
    cfg = load_config(os.path.join(ROOT, 'configs', 'pseudo_fb33_superpixel.yaml'),
                      ['train.epochs=3', 'data.val_fraction=0.1'])
    assert cfg['model']['head']['type'] == 'pseudo'                  # from pseudo_fb33.yaml
    assert cfg['train']['multiscale'] == [1] and cfg['train']['lr'] == 1e-4   # base + parent
    assert cfg['train']['epochs'] == 3 and cfg['data']['val_fraction'] == 0.1
    assert set(cfg['loss']) == {'pseudo_ce', 'dice'}                 # losses are not inherited from the base


def test_parse_logs(tmp_path):
    lines = []
    for ep, dices in [(1, [0.9, 0.9, 0.8, 0.7, 0.6]), (2, [0.95, 0.9, 0.85, 0.75, 0.7])]:
        for ds, d in zip(['CVC-ClinicDB', 'Kvasir', 'CVC-300', 'CVC-ColonDB', 'ETIS-LaribPolypDB'], dices):
            lines.append(f'[2026-01-01 10:00:00 AM-train.py-INFO:epoch: {ep}, dataset: {ds}, dice: {d}]')
        lines.append(f'[2026-01-01 10:00:00 AM-train.py-INFO:Validation dice score: {sum(dices) / 5}]')
    (tmp_path / 'log.log').write_text('\n'.join(lines))
    runs = parse(str(tmp_path / 'log.log'))
    assert len(runs) == 1 and sorted(runs[0]['epochs']) == [1, 2]
    assert runs[0]['epochs'][2]['Kvasir'] == 0.9


def test_trainer_end_to_end_on_cpu(tiny_data, tmp_path):
    """Two tiny epochs on random images: seeding, validation split, checkpointing and a parseable log."""
    from protoseg_polyp.engine.trainer import train
    from protoseg_polyp.models import build_model
    cfg = load_config(os.path.join(ROOT, 'configs', 'pvt_proto_fb88.yaml'), [
        'model.encoder.pretrained=null', f'data.train_root={tiny_data / "TrainDataset"}',
        f'data.test_root={tiny_data / "TestDataset"}', 'data.val_fraction=0.25', 'train.epochs=2',
        'train.batchsize=3', 'train.img_size=64', 'train.multiscale=[0.75,1]', 'train.save_epochs=[2]'])
    out = tmp_path / 'run'
    best = train(build_model(cfg['model']), cfg, str(out), device='cpu')
    assert 0 <= best <= 1
    assert {'best.pth', 'check.pth', 'epoch_2.pth', 'train.log', 'prototype_diagnostics.png'} <= set(os.listdir(out))
    runs = parse(str(out / 'train.log'))
    assert sorted(runs[0]['epochs']) == [1, 2]
    sd = torch.load(out / 'best.pth', map_location='cpu')
    assert any(k.startswith('encoder.') for k in sd)
