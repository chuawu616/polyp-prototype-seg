#!/usr/bin/env bash
# Follow-up experiments (Sep 2026), run sequentially on one GPU. Safe to re-run: finished runs (runs/followup/<name>/DONE)
# are skipped and an unfinished run is moved aside (<name>.incomplete-<time>) and restarted.
#
#   GPU=0 bash scripts/run_followup.sh            # DRY_RUN=1 prints the commands only
#
# 1. diagnostics (~3 h): V0 with the original vs. corrected PPD; pseudo-label + superpixels at the original (50 px at 352,
#    ~3 px at d1) vs. the intended granularity (800 px at 352, ~50 px at d1)
# 2. seeds (~16 h): linear head vs. vanilla prototypes fb88, three seeds each, checkpoint selected on a 10 % validation split
set -u
cd "$(dirname "$0")/.."
GPU=${GPU:-0}
PY=${PY:-python}
OUT=runs/followup
SP50=${SP50:-runs/cache/superpixels_pps50}
SP800=${SP800:-runs/cache/superpixels_pps800}
COMMON="data.val_fraction=0.1"
mkdir -p "$OUT"

run() {  # run <name> <config> [--set overrides...]
  local name=$1 cfg=$2; shift 2
  local dir="$OUT/$name"
  if [ -f "$dir/DONE" ]; then echo "[skip] $name"; return; fi
  if [ -d "$dir" ]; then mv "$dir" "$dir.incomplete-$(date +%m%d-%H%M)"; fi
  echo "[$(date '+%F %T')] start $name"
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "  $PY tools/train.py $cfg --out $dir --set $COMMON $*"; return; fi
  mkdir -p "$dir"
  if CUDA_VISIBLE_DEVICES=$GPU $PY tools/train.py "$cfg" --out "$dir" --set $COMMON "$@" > "$dir/stdout.log" 2>&1; then
    date '+%F %T' > "$dir/DONE"; echo "[$(date '+%F %T')] done  $name"
  else
    echo "[$(date '+%F %T')] FAILED $name (see $dir/stdout.log)"
  fi
}

superpixels() {  # superpixels <dir> <pixels_per_sp>; skipped if the folder already holds one map per training image
  local n_img n_sp
  n_img=$(ls data/polyp/TrainDataset/images | wc -l)
  n_sp=$(ls "$1" 2>/dev/null | grep -c '\.npy$')
  if [ -f "$1/DONE" ] || [ "$n_sp" -ge "$n_img" ]; then echo "[skip] superpixels $1 ($n_sp maps)"; return; fi
  echo "[$(date '+%F %T')] superpixels -> $1 (pixels_per_sp=$2)"
  [ "${DRY_RUN:-0}" = 1 ] && return
  $PY tools/precompute_superpixels.py --train_root data/polyp/TrainDataset --save_dir "$1" --pixels_per_sp "$2" \
    && date '+%F %T' > "$1/DONE"
}

# ---- 1. diagnostics
run v0_ppd_original  configs/nonlearnable_v0_fb33.yaml   train.seed=0 "train.save_epochs=[1,5,20,100]"
run v0_ppd_corrected configs/nonlearnable_v0_fb33.yaml   train.seed=0 "train.save_epochs=[1,5,20,100]" loss.ppd.legacy_scaled=false
superpixels "$SP50" 50
superpixels "$SP800" 800
run pseudo_sp50      configs/pseudo_fb33_superpixel.yaml train.seed=0 "train.save_epochs=[1,5,10,30]" data.superpixel_root=$SP50
run pseudo_sp800     configs/pseudo_fb33_superpixel.yaml train.seed=0 "train.save_epochs=[1,5,10,30]" data.superpixel_root=$SP800

# ---- 2. seeds (interleaved, so a partial run still gives paired comparisons)
for s in 0 1 2; do
  run linear_s$s     configs/pvt_linear.yaml     train.seed=$s
  run proto_fb88_s$s configs/pvt_proto_fb88.yaml train.seed=$s
done
echo "[$(date '+%F %T')] all done — summarise with: $PY tools/summarize_runs.py $OUT"
