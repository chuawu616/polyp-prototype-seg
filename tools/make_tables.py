"""Build the Markdown result tables of README.md / docs/results.md from docs/results/reevaluation.csv.

  python tools/make_tables.py            # prints the tables
"""
import csv
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, 'docs', 'results', 'reevaluation.csv')
DS = ['CVC-ClinicDB', 'Kvasir', 'CVC-300', 'CVC-ColonDB', 'ETIS-LaribPolypDB']


def label(ck):
    """(stage, method, variant) for a checkpoint folder of the original code."""
    name = ck.split('/')[-1]
    if ck == 'cascade/models/polyp/baseline':
        return 2, 'Linear head (reference)', 'Dec 2025, earlier TA decoder variant (base+CRFB+LGAG)'
    if ck.startswith('cascade/'):
        if 'kl_loss' in name:
            w = re.search(r'kl_loss_([\d.]+)', name).group(1)
            fb = re.search(r'fb_(\d+)', name).group(1)
            if 'pretrain' in name:
                return 2, 'Learnable prototype + contrastive pretrain', f'fb{fb} + KL w={w}'
            return 2, 'Learnable prototype + KL', f'fb{fb}, w={w}' + (' (v3)' if name.endswith('v3') else '')
        if 'hybrid' in name:
            return 4, 'Specialise (hybrid)', 'fhb' + re.search(r'fhb_(\d+)', name).group(1) + \
                (' warm' if 'warm' in name else '')
        if 'fhb' in name:
            v = 'fhb' + re.search(r'fhb_(\d+)', name).group(1)
            for tag in ('warm_10', 'aug', 'orth_loss', 'uncertainty_gate'):
                if tag in name:
                    v += ' ' + {'warm_10': 'warm10', 'orth_loss': '+orth', 'aug': '+aug',
                                'uncertainty_gate': '+uncert. gate'}[tag]
            return 4, 'Specialise', v
        if 'pretrain' in name:
            return 2, 'Learnable prototype + contrastive pretrain', 'fb44'
        fb = re.search(r'fb_(\d+)', name).group(1)
        return 2, 'Learnable prototype', 'fb' + fb + (' (re-run)' if name.endswith('v2') else '')
    if ck.startswith('dino_v3/'):
        arch = {'vits16plus': 'ViT-S+', 'vitb16': 'ViT-B', 'vitl16': 'ViT-L'}[ck.split('/')[3]]
        if '/polyp/' in ck:
            concat = 'concat' in name
            ft = name.endswith('fine')
            return 3, 'DINOv3 + MLP', f'{arch}, {"blocks 2..12" if concat else "last block"}, ' + \
                ('fine-tuned' if ft else 'frozen')
        if 'specialize' in name:
            return 4, 'DINOv3 specialise', 'fhb' + re.search(r'fhb_(\d+)', name).group(1) + \
                (' warm10' if 'warm' in name else '')
        fb = re.search(r'fb_(\d+)', name).group(1)
        concat = 'concat' in name
        ft = 'fine' in name or 'smooth' in name
        return 3, 'DINOv3 + prototype', f'fb{fb}, {"blocks 2..12" if concat else "last block"}, ' + \
            ('fine-tuned' if ft else 'frozen') + (' (smooth)' if 'smooth' in name else '')
    if ck.startswith('non_learnable/'):
        if name == 'baseline_100epoch':
            return 2, 'Linear head (reference)', 'May 2026, EMCAD decoder (same as all prototype runs)'
        if name == 'baseline_dinov3':
            return 3, 'DINOv3 + linear', 'ViT-S+, blocks 2..12, fine-tuned'
        if name.startswith('prototypev3'):
            v = 'x4 node' if name.endswith('x4') else 'dd4 node'
            v += ', ImageNet init' if 'wo_pretrained' in name else ', pretrained init'
            run = re.search(r'_(\d)$', name)
            return 5, 'Non-learnable V3', v + (f' (run {int(run.group(1)) + 1})' if run and 'ppc_1_' in name else '')
        if name.startswith('prototype_v0'):
            return 5, 'Non-learnable V0', 'fb33' if 'fb33' in name else 'fb55'
        if 'pseudo' in name:
            return 5, 'Pseudo-label', {'prototype_pseudo_fb33': 'fb33', 'prototype_pseudo_fb33_pretrained':
                                       'fb33, pretrained init', 'prototype_pseudo_fb33_sp': 'fb33 + superpixel'}[name]
    raise ValueError(ck)


def rows():
    out = []
    for r in csv.DictReader(open(CSV)):
        stage, method, variant = label(r['checkpoint'])
        out.append(dict(r, stage=stage, method=method, variant=variant))
    return sorted(out, key=lambda r: (r['stage'], r['method'], -float(r['mDice'])))


def full_table(rs):
    lines = ['| Stage | Method | Variant | ClinicDB | Kvasir | CVC-300 | ColonDB | ETIS | In | Out | **mDice** | mIoU '
             '| orig. mDice | verified | checkpoint |', '|' + '---|' * 15]
    for r in rs:
        v = '✓' if r['verified'] == 'True' else '✗'
        lines.append(f"| {r['stage']} | {r['method']} | {r['variant']} | " +
                     ' | '.join(f"{float(r[d]):.3f}" for d in DS) +
                     f" | {float(r['in_domain']):.3f} | {float(r['out_domain']):.3f} | **{float(r['mDice']):.3f}** "
                     f"| {float(r['mIoU']):.3f} | {float(r['legacy_mDice']):.3f} | {v} | `{r['checkpoint']}` |")
    return '\n'.join(lines)


def summary_table(rs):
    """Best checkpoint per method + spread over its variants."""
    by = {}
    for r in rs:
        by.setdefault((r['stage'], r['method']), []).append(r)
    lines = ['| Stage | Method | Best variant | **mDice** | In-domain | Out-of-domain | Range over variants (n) |',
             '|---|---|---|---|---|---|---|']
    for (stage, method), group in sorted(by.items()):
        best = max(group, key=lambda r: float(r['mDice']))
        vals = [float(r['mDice']) for r in group]
        rng = f'{min(vals):.3f}–{max(vals):.3f} ({len(vals)})' if len(vals) > 1 else '–'
        mark = '' if best['verified'] == 'True' else ' †'
        lines.append(f"| {stage} | {method} | {best['variant']}{mark} | **{float(best['mDice']):.3f}** | "
                     f"{float(best['in_domain']):.3f} | {float(best['out_domain']):.3f} | {rng} |")
    return '\n'.join(lines)


if __name__ == '__main__':
    rs = rows()
    print(summary_table(rs))
    print()
    print(full_table(rs))
