"""YAML configs with single inheritance: a config may name a `base:` file whose values it overrides."""
import copy
import os

import yaml


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path, overrides=()):
    """overrides: ['train.epochs=1', 'data.test_root=/x'] (values parsed as YAML)."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if 'base' in cfg:
        cfg = _merge(load_config(os.path.join(os.path.dirname(path), cfg.pop('base'))), cfg)
    for item in overrides:
        key, val = item.split('=', 1)
        node = cfg
        *parents, leaf = key.split('.')
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return cfg
