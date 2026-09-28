# Prototype-based Polyp Segmentation

[![tests](https://github.com/chuawu616/polyp-prototype-seg/actions/workflows/tests.yml/badge.svg)](https://github.com/chuawu616/polyp-prototype-seg/actions/workflows/tests.yml)

**Modelling intra-class variation with sub-class prototypes — a one-year undergraduate research project**

NYCU EE · advisor Prof. 黃俊達 · TA 陳泳翰 · Sep 2025 – Jun 2026

## TL;DR

- **Question.** Polyps and the colon wall vary a lot in appearance. Can a segmentation head that represents each class
  with several *sub-class prototypes*, instead of one linear boundary, capture this variation and segment better?
- **What I built.** Five generations of prototype heads on a PVTv2 + EMCAD baseline: learnable prototypes,
  easy/hard specialisation, a DINOv3 backbone, non-learnable prototypes (Sinkhorn assignment + EMA, after ProtoSeg)
  and a prototype pseudo-label teacher with superpixels.
- **Outcome.** The best prototype head reaches **0.876 mDice**, above EMCAD's published 0.866 and on par with the
  linear head I re-trained (0.872–0.876). Across runs and variants the prototype heads did not consistently exceed
  the linear baseline.
- **Main finding.** The sub-classes the prototypes discover follow *geometry* — body vs. rim, concentric rings,
  horizontal bands — rather than tissue appearance. Binary masks define the boundary between classes but not the
  structure inside them, so a prototype head can only partition what the features already separate; a semantic
  signal from outside the binary labels (e.g. self-supervised features) is the natural next step.
- **Engineering.** Five research code bases merged into one package; all 70 surviving checkpoints re-evaluated under
  one protocol and matched, epoch by epoch, to their original training logs.

<table>
<tr>
<td width="50%"><img src="docs/figures/stage2_subclass_maps.jpg"><br><sub><b>Learnable prototypes</b> (1–8 per class):
extra prototypes split the polyp into body vs. rim.</sub></td>
<td width="50%"><img src="docs/figures/stage5_v2_umap.png"><br><sub><b>Feature space</b> (UMAP): the five FG prototypes
land on one point — polyp features form a single blob.</sub></td>
</tr>
<tr>
<td><img src="docs/figures/stage5_v0_rings.jpg"><br><sub><b>Non-learnable prototypes</b>: sub-classes become
concentric rings, identical on every image.</sub></td>
<td><img src="docs/figures/stage5_v3_init_comparison.jpg"><br><sub><b>Initialisation matters</b>: with binary-pretrained
features the sub-class map is binary from epoch 1 (left); from ImageNet features, sub-classes appear and then merge (right).</sub></td>
</tr>
</table>

## Results (unified re-evaluation of all 70 surviving checkpoints)

mDice = mean Dice over CVC-ClinicDB, Kvasir, CVC-300, CVC-ColonDB and ETIS (EMCAD paper: 0.866). One decision rule for every model; see [docs/results.md](docs/results.md) for the protocol and all 70 rows.

| Stage | Method | Best variant | **mDice** | In-domain | Out-of-domain | Range over variants (n) |
|---|---|---|---|---|---|---|
| 2 | Learnable prototype | fb88 | **0.876** | 0.929 | 0.841 | 0.865–0.876 (10) |
| 2 | Learnable prototype + KL | fb33, w=0.1 | **0.870** | 0.934 | 0.827 | 0.863–0.870 (7) |
| 2 | Learnable prototype + contrastive pretrain | fb44 | **0.774** | 0.873 | 0.707 | 0.764–0.774 (3) |
| 2 | Linear head (reference) | Dec 2025, earlier TA decoder variant (base+CRFB+LGAG) | **0.876** | 0.937 | 0.836 | 0.872–0.876 (2) |
| 3 | DINOv3 + MLP | ViT-S+, last block, fine-tuned † | **0.869** | 0.934 | 0.826 | 0.785–0.869 (6) |
| 3 | DINOv3 + linear | ViT-S+, blocks 2..12, fine-tuned | **0.869** | 0.936 | 0.824 | – |
| 3 | DINOv3 + prototype | fb44, blocks 2..12, fine-tuned | **0.870** | 0.931 | 0.830 | 0.794–0.870 (9) |
| 4 | DINOv3 specialise | fhb424 | **0.870** | 0.935 | 0.827 | 0.865–0.870 (6) |
| 4 | Specialise | fhb524 warm10 | **0.874** | 0.938 | 0.832 | 0.867–0.874 (11) |
| 4 | Specialise (hybrid) | fhb424 warm | **0.870** | 0.933 | 0.827 | 0.864–0.870 (4) |
| 5 | Non-learnable V0 | fb33 | **0.845** | 0.918 | 0.795 | 0.838–0.845 (2) |
| 5 | Non-learnable V3 | dd4 node, pretrained init | **0.869** | 0.932 | 0.828 | 0.856–0.869 (6) |
| 5 | Pseudo-label | fb33 | **0.857** | 0.910 | 0.821 | 0.847–0.857 (3) |

† no training log to verify against. Differences below ≈ 0.01 are within run-to-run noise (the same fb88 configuration trained twice: 0.876 and 0.866).

## The five stages

| Stage | Period | Idea | Best mDice | Takeaway |
|---|---|---|---|---|
| 0 | Sep–Oct 2025 | Self-supervised few-shot (SSL-ALPNet) with superpixel pseudo-labels | 0.11 | A support-conditioned model cannot create sub-classes the task never defines |
| 1 | Nov 2025 | Three-stage pipeline built from scratch: pixel-contrastive → KMeans sub-labels → prototypes | ≈0.80* | Too many coupled components to isolate causes → restarted from the TA's code with minimal changes |
| 2 | Dec 2025 | Learnable prototypes replace the 1×1 head | 0.876 | On par with the linear head; prototype usage is highly skewed; enforcing balance (KL) fragments the masks |
| 3 | Mar 2026 | DINOv3 backbone (ViT-S+/B/L, frozen and fine-tuned) | 0.870 | Stronger generic features did not diversify the prototypes; fine-tuning drives them to cos 0.97–0.99 |
| 4 | Mar–Apr 2026 | Easy / hard prototype specialisation, hybrid image-conditioned prototypes | 0.874 | Within run-to-run variation; both backbones struggle on the same hard images |
| 5 | Apr–Jun 2026 | Non-learnable prototypes (ProtoSeg): Sinkhorn assignment + EMA; V2 → V3 → V0 → pseudo-label → superpixel | 0.869 | A balanced assignment is not a semantic one: without appearance cues, Sinkhorn partitions by position |

\* own evaluation script, not comparable. The full account with figures: **[docs/journey.md](docs/journey.md)**.

## What this repository demonstrates

- **Verified refactor.** Five research code bases (≈ 70 model variants) merged into one package
  (`encoder → neck → head`, 8 head types). All 70 surviving checkpoints load strictly into it, and for every
  checkpoint with a training log the refactored evaluation reproduces the logged per-dataset Dice of a specific
  epoch to < 1e-4 ([docs/results/reevaluation.csv](docs/results/reevaluation.csv)).
- **One evaluation protocol.** The original scripts mixed soft Dice, hard Dice and a foreground-only decision
  rule. Re-scoring everything with one rule shifts some models (non-learnable V3 by up to +4 points) but none of
  the conclusions.
- **Issues identified during re-implementation**, documented in [journey.md](docs/journey.md#diagnosis-why-the-sub-classes-stayed-geometric):
  a temperature-scaling mismatch in the PPD loss, a test-time decision rule that differed from training,
  superpixels 16× finer than intended, an inactive edge weighting in the inherited structure loss, and a KMeans
  initialisation that was silently skipped.
- **Careful statistics.** A re-run of the best prototype configuration (fb88: 0.876) reached 0.866, and the linear
  head reaches 0.872–0.876, so differences below ≈ 0.01 are treated as run-to-run variation rather than method gains.

## Repository layout

```
protoseg_polyp/
  models/       encoders.py (PVTv2+EMCAD, DINOv3) · heads.py (linear, MLP, prototype, specialise/hybrid)
                nonlearnable.py (V2, V3, V0, pseudo-label) · pvtv2.py · __init__.py (builder, legacy loader)
                third_party/ (EMCAD decoder, downloaded by tools/fetch_third_party.py)
  engine/       trainer.py · evaluate.py (unified + legacy protocols) · losses.py · trackers.py (collapse curves)
  data/         PraNet-split datasets, optional superpixel maps, train/val split
  ops.py        Sinkhorn-Knopp, EMA
configs/        one YAML per experiment (inherits _base*.yaml)
tools/          train.py · test.py · analyze_features.py · precompute_superpixels.py · extract_kmeans_centers.py
                fetch_third_party.py · parse_logs.py · reevaluate_legacy.py · make_tables.py
tests/          CPU unit tests (models, legacy loading, losses, metrics, data, trainer); run by GitHub Actions
docs/           journey.md · results.md · results/*.csv · figures/
archive/        original research code of every stage (not maintained, see archive/README.md)
```

## Reproducing

```bash
conda create -n protoseg python=3.10 -y && conda activate protoseg
pip install -r requirements.txt
python tools/fetch_third_party.py      # downloads the EMCAD decoder (research license, not redistributable)
```

**Data.** The standard PraNet split (train: Kvasir + CVC-ClinicDB, 1,450 images; test: CVC-ClinicDB, Kvasir,
CVC-300, CVC-ColonDB, ETIS-LaribPolypDB, 798 images) from the
[PraNet repository](https://github.com/DengPingFan/PraNet), arranged as
`data/polyp/{TrainDataset,TestDataset/<name>}/{images,masks}/`.

**Weights.** ImageNet PVTv2-b2 and DINOv3 backbones go in `weights/` (see [weights/README.md](weights/README.md)).
Trained checkpoints are not included in this repository (see [checkpoints/README.md](checkpoints/README.md)).

```bash
# evaluate (checkpoints of the original research code are converted automatically)
python tools/test.py configs/pvt_proto_fb88.yaml checkpoints/pvt_proto_fb88.pth --save_dir preds/fb88

# train; for new experiments hold out 10 % of the training images for checkpoint selection
python tools/train.py configs/pseudo_fb33.yaml --out runs/pseudo_fb33 --set data.val_fraction=0.1

# sub-class maps + PCA of every feature node, best/worst cases or across epochs
python tools/analyze_features.py configs/pseudo_fb33.yaml checkpoints/pseudo_fb33.pth --worst 8 --out figs/worst.jpg

# inputs for the Stage-5 variants
python tools/precompute_superpixels.py --train_root data/polyp/TrainDataset
python tools/extract_kmeans_centers.py configs/pvt_linear.yaml checkpoints/pvt_linear.pth --node dd4 --out assets/kmeans/dd4_ch512_m5.pth
```

Everything falls back to CPU when no GPU is present (evaluating a PVTv2 model on all 798 test images takes about two
minutes); training needs a GPU. Configs use paths relative to the repository root; `--set KEY=VALUE` overrides any
config entry.

**Verification status.**

| Part | How it was verified |
|---|---|
| Models + evaluation | all 70 original checkpoints load strictly; 66 reproduce their logged per-dataset Dice to < 1e-4 (the other 4 have no log or lost code) |
| Third-party replacements | official PVTv2 and unmodified EMCAD give bit-identical features to the original code (max abs. difference 0.0) |
| Training loop | unit tests: one optimisation step for every config, EMA-only prototype updates, and a two-epoch CPU run on synthetic images (seeding, validation split, checkpointing, log format); **not yet used to retrain a model on the real data** |

**Tests.** `pip install -r requirements-dev.txt && pytest` runs 42 CPU tests in about 30 s without data or
weights (DINOv3 tests are skipped unless `PROTOSEG_TEST_DINOV3=1`, as they download the model code); the same suite
runs on every push via GitHub Actions.

**Evaluation protocol**, naming and every number: [docs/results.md](docs/results.md). The original experiments
selected checkpoints on the test sets (no validation split), so their numbers are optimistic; `data.val_fraction`
and `train.seed` (see `configs/_base.yaml`) make new runs clean and repeatable.

## License and acknowledgements

The code written for this repository is released under the [MIT License](LICENSE). Third-party components keep
their own licenses — see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). In particular, the EMCAD decoder is
used under the UT Austin Research License (academic, non-commercial use) and is downloaded rather than included.

The PVTv2-b2 + EMCAD baseline code was provided by the TA, based on
[EMCAD](https://github.com/SLDGroup/EMCAD) (Rahman et al., CVPR 2024) and [PVT](https://github.com/whai362/PVT)
(Wang et al., CVMJ 2022); non-learnable prototypes follow [ProtoSeg](https://github.com/tfzhou/ProtoSeg)
(Zhou et al., *Rethinking Semantic Segmentation: A Prototype View*, CVPR 2022);
Stage 0 builds on [SSL-ALPNet](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation)
(Ouyang et al., TMI 2022); [DINOv3](https://github.com/facebookresearch/dinov3) (Meta).
