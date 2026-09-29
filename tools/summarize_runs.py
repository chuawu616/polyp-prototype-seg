"""Summarise runs trained with a validation split (e.g. scripts/run_followup.sh).

For every run the epoch with the best validation Dice is selected (first maximum, as the trainer does) and the test
scores of that epoch are reported; runs named <group>_s<seed> are aggregated to mean ± std.

  python tools/summarize_runs.py runs/followup [--out docs/results/followup.md]
"""
import argparse
import glob
import os
import re
from collections import OrderedDict, defaultdict

import numpy as np

DS = ['CVC-ClinicDB', 'Kvasir', 'CVC-300', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
RE_DS = re.compile(r'epoch: (\d+), dataset: ([\w\-]+), dice: ([\d.eE\-]+)')
RE_VAL = re.compile(r'epoch: (\d+), val dice: ([\d.eE\-]+)')


def read(log):
    test, val = defaultdict(dict), {}
    for line in open(log, errors='ignore'):
        m = RE_DS.search(line)
        if m:
            test[int(m.group(1))][m.group(2)] = float(m.group(3))
        m = RE_VAL.search(line)
        if m:
            val[int(m.group(1))] = float(m.group(2))
    return test, val


def summarise(run_dir):
    test, val = read(os.path.join(run_dir, 'train.log'))
    epochs = [e for e in sorted(val) if all(d in test[e] for d in DS)]
    if not epochs:
        return None
    best = max(epochs, key=lambda e: (val[e], -e))       # first maximum
    t = test[best]
    return OrderedDict(epoch=best, epochs=len(epochs), val=val[best], **{d: t[d] for d in DS},
                       mDice=np.mean([t[d] for d in DS]), in_domain=np.mean([t[d] for d in DS[:2]]),
                       out_domain=np.mean([t[d] for d in DS[2:]]), done=os.path.exists(os.path.join(run_dir, 'DONE')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    runs = OrderedDict()
    for d in sorted(glob.glob(os.path.join(args.root, '*'))):
        if os.path.isdir(d) and '.incomplete' not in d and os.path.exists(os.path.join(d, 'train.log')):
            r = summarise(d)
            if r:
                runs[os.path.basename(d)] = r
    lines = ['| run | best epoch (of) | val Dice | ' + ' | '.join(DS) + ' | in | out | **test mDice** | finished |',
             '|' + '---|' * (len(DS) + 7)]
    for name, r in runs.items():
        lines.append(f"| {name} | {r['epoch']} ({r['epochs']}) | {r['val']:.4f} | " + ' | '.join(f'{r[d]:.4f}' for d in DS) +
                     f" | {r['in_domain']:.4f} | {r['out_domain']:.4f} | **{r['mDice']:.4f}** | {'yes' if r['done'] else 'no'} |")
    groups = defaultdict(list)
    for name, r in runs.items():
        if r['done']:                                    # unfinished runs are listed above but not aggregated
            groups[re.sub(r'_s\d+$', '', name)].append(r)
    lines += ['', 'Groups aggregate finished runs only.', '', '| group | n | val Dice | in-domain | out-of-domain | **test mDice** |', '|---|---|---|---|---|---|']
    for g, rs in groups.items():
        def ms(k):
            v = np.array([r[k] for r in rs])
            return f'{v.mean():.4f} ± {v.std(ddof=1):.4f}' if len(v) > 1 else f'{v.mean():.4f}'
        lines.append(f'| {g} | {len(rs)} | {ms("val")} | {ms("in_domain")} | {ms("out_domain")} | **{ms("mDice")}** |')
    text = '\n'.join(lines)
    print(text)
    if args.out:
        with open(args.out, 'w') as f:
            f.write('# Follow-up runs (checkpoint selected on a 10 % validation split)\n\n' + text + '\n')


if __name__ == '__main__':
    main()
