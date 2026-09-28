"""Rebuild the experiment table from raw training logs.

Every training script (inherited from the EMCAD/CASCADE polyp code) logs
    epoch: N, dataset: <name>, dice: <float>     (x5 test sets)
    Validation dice score: <float>               (image-weighted over 798 images)
once per epoch. `best.pth` is selected by mDice = unweighted mean of the five
per-dataset dice, which is only printed to stdout, so it is recomputed here.
"""
import argparse
import csv
import os
import re
from collections import OrderedDict

DATASETS = ["CVC-ClinicDB", "Kvasir", "CVC-300", "CVC-ColonDB", "ETIS-LaribPolypDB"]
RE_DS = re.compile(r"^\[(?P<ts>[\d\- :]+ [AP]M)-(?P<script>[^-\s]+\.py)-INFO:epoch: (?P<ep>\d+), dataset: (?P<ds>[\w\-]+), dice: (?P<dice>[\d.eE\-]+)")
RE_VAL = re.compile(r"Validation dice score: (?P<m>[\d.eE\-]+)")
RE_TOTAL = re.compile(r"Epoch \[\d+/(?P<tot>\d+)\]")


def parse(path):
    """Return a list of runs (a log may contain restarted runs)."""
    runs, cur, last_ep, script, ts0, planned = [], None, None, None, None, None
    with open(path, errors="ignore") as f:
        for line in f:
            m = RE_TOTAL.search(line)
            if m and planned is None:
                planned = int(m.group("tot"))
            m = RE_DS.search(line)
            if m:
                ep = int(m.group("ep"))
                if cur is None or (last_ep is not None and ep < last_ep):
                    cur = {"epochs": OrderedDict(), "script": m.group("script"), "start": m.group("ts")}
                    runs.append(cur)
                last_ep = ep
                cur["epochs"].setdefault(ep, {})[m.group("ds")] = float(m.group("dice"))
                continue
            m = RE_VAL.search(line)
            if m and cur is not None and last_ep is not None:
                cur["epochs"][last_ep]["mdice"] = float(m.group("m"))
    for r in runs:
        r["planned"] = planned
    return runs


def summarize(name, path, run, idx, nruns):
    eps = {e: d for e, d in run["epochs"].items() if all(ds in d for ds in DATASETS)}
    if not eps:
        return None
    for d in eps.values():
        d["mean5"] = sum(d[ds] for ds in DATASETS) / len(DATASETS)
    best_ep = max(eps, key=lambda e: eps[e]["mean5"])
    last_ep = max(eps)
    row = OrderedDict(
        experiment=name + (f"#run{idx + 1}" if nruns > 1 else ""),
        stage=path.split(os.sep)[0],
        script=run["script"],
        start=run["start"],
        epochs_logged=len(eps),
        best_epoch=best_ep,
        best_mdice=round(eps[best_ep]["mean5"], 4),
        weighted_dice_at_best=round(eps[best_ep]["mdice"], 4) if "mdice" in eps[best_ep] else "",
        last_mdice=round(eps[last_ep]["mean5"], 4),
    )
    for ds in DATASETS:
        v = eps[best_ep].get(ds)
        row[ds] = round(v, 4) if v is not None else ""
    row["log"] = path
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--dirs", nargs="+", default=["cascade", "dino_v3", "non_learnable"])
    ap.add_argument("--out", default="experiments.csv")
    args = ap.parse_args()

    rows, seen = [], {}
    for d in args.dirs:
        for dp, dns, fns in os.walk(os.path.join(args.root, d)):
            dns[:] = [x for x in dns if x != ".ipynb_checkpoints" and x != "ProtoSeg"]
            for fn in sorted(fns):
                if not fn.endswith(".log"):
                    continue
                full = os.path.join(dp, fn)
                rel = os.path.relpath(full, args.root)
                name = re.sub(r"^train_log_", "", fn[:-4])
                runs = parse(full)
                for i, r in enumerate(runs):
                    row = summarize(name, rel, r, i, len(runs))
                    if row:
                        rows.append(row)
    rows.sort(key=lambda r: (r["stage"], r["start"]))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} runs -> {args.out}")


if __name__ == "__main__":
    main()
