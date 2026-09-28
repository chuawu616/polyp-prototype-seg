# Archive: original research code

These folders are snapshots of the code as it was written during the project, kept so the history in
[`docs/journey.md`](../docs/journey.md) can be traced back to real code. They are **not maintained** and
contain hard-coded paths of the lab server (`/home/U116med/...`); in this public copy, third-party files are
removed (see the end of this page). Everything reproducible lives in
`protoseg_polyp/`, `configs/` and `tools/`; the refactor was verified by loading the original checkpoints
and matching their logged scores (see [`docs/results/reevaluation.csv`](../docs/results/reevaluation.csv)).

| Folder | Period | What it is |
|---|---|---|
| `0_ssl_fss/` | Sep–Oct 2025 | Additions to [SSL-ALPNet](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation) (upstream commit `a3029d8`) for polyp data: superpixel pseudo-label generators (Felzenszwalb / SLIC / watershed notebooks), polyp dataset, training/validation scripts. Files that modify upstream code are stored as `patches_vs_upstream/*.patch`; upstream code itself is not redistributed. |
| `1_pvt_kmeans/` | Nov–Dec 2025 | Self-built three-stage pipeline: pixel-contrastive pre-training (`train_stage1.py`), KMeans sub-labels + sub-label contrastive (`generate_aligned_sublabels.py`, `train_stage2.py`), prototype segmentation (`train_stage3*.py`, decoder variants ASPP / CARA / EMCAD). `runs/` keeps the Sacred `config.json` / `metrics.json` of each run; `prototype_analysis/` and `sublabel_generation_check/` hold the figures discussed in the Nov 2025 reports. Evaluated with its own script, so scores are not comparable with the main table. |
| `original_code/2_cascade/` | Dec 2025 – Apr 2026 | TA's PVTv2 + EMCAD code with the author's learnable-prototype, KL, specialise and hybrid heads; analysis notebooks (utilisation, PCA, hard samples). The TA's decoder ablations (`unused lib/decoders_*.py`, removed here) matter for one result: the Dec 2025 linear baseline checkpoint was trained with the `base+CRFB+LGAG` variant, not the final EMCAD decoder. |
| `original_code/3_dino_v3/` | Mar–Apr 2026 | DINOv3 backbones (linear / MLP / prototype / specialise / adaptive-gate heads), worst-case and model-comparison scripts. |
| `original_code/4_non_learnable/` | Apr–Jun 2026 | Non-learnable prototype versions V2 (`train_prototype_2.py`), V3 (`train_prototype_4.py`), V0, pseudo-label (+superpixel), KMeans / superpixel pre-computation, PCA-per-node and epoch-evolution analysis scripts. |

Known issues in the original code (all documented in `docs/journey.md`):

- V0/V3 apply PPD to temperature-scaled logits, i.e. PPD pulls the pixel-prototype cosine toward 0.1.
- V0/V3 test functions threshold the FG similarity alone (BG ignored), unlike the training decision rule.
- `structure_loss` uses the deprecated `reduce='none'`, so its BCE edge weighting is inactive (inherited from PraNet).
- `pseudo_fb33_pretrained` requested 5-prototype KMeans centres for a 3-prototype model; the shape check silently fell back to random initialisation.
- `precompute_superpixels.py` computed `area / pixels_per_sp` at 352×352 although the intended count was per d1 (88×88) pixel, giving ≈16× more superpixels (≈3 d1 pixels each).
- The forward pass of the `uncertainty_gate` specialise variant is not in any surviving file; only its checkpoint remains (evaluated with the plain specialise forward it does not reproduce its log).

## Files removed from this public copy

The files below were copies (or lightly modified copies) of third-party code that may not be redistributed:
EMCAD is released under the UT Austin Research License, which prohibits redistribution of the software or any
portion of it, and Polyp-PVT's repository carries no license. They were removed here; the author keeps the
complete snapshot privately. The scripts in this archive therefore no longer import as-is — they were never
meant to run from here (hard-coded server paths); the maintained, runnable implementation is `protoseg_polyp/`,
which downloads the unmodified EMCAD decoder with `tools/fetch_third_party.py`.

| Removed file | Upstream | Relation |
|---|---|---|
| `1_pvt_kmeans/models/decoders_emcad.py` | EMCAD [`lib/decoders.py`](https://github.com/SLDGroup/EMCAD/blob/26c9c31f73/lib/decoders.py) | near-verbatim copy |
| `1_pvt_kmeans/models/pvtv2.py` | Polyp-PVT [`lib/pvtv2.py`](https://github.com/DengPingFan/Polyp-PVT/blob/main/lib/pvtv2.py) | near-verbatim copy |
| `original_code/2_cascade/lib/decoders.py` | EMCAD [`lib/decoders.py`](https://github.com/SLDGroup/EMCAD/blob/26c9c31f73/lib/decoders.py) | near-verbatim copy |
| `original_code/2_cascade/lib/networks.py` | EMCAD [`lib/networks.py`](https://github.com/SLDGroup/EMCAD/blob/26c9c31f73/lib/networks.py) | wraps the EMCAD decoder (PVT_CASCADE) |
| `original_code/2_cascade/lib/pvtv2.py` | Polyp-PVT [`lib/pvtv2.py`](https://github.com/DengPingFan/Polyp-PVT/blob/main/lib/pvtv2.py) | near-verbatim copy |
| `original_code/2_cascade/unused lib/decoders1.py` | EMCAD [`lib/decoders.py`](https://github.com/SLDGroup/EMCAD/blob/26c9c31f73/lib/decoders.py) | near-verbatim copy |
| `original_code/2_cascade/unused lib/decoders_1_base.py` | EMCAD [`lib/decoders.py`](https://github.com/SLDGroup/EMCAD/blob/26c9c31f73/lib/decoders.py) | modified copy (TA decoder ablation) |
| `original_code/2_cascade/unused lib/decoders_2_base-crfb.py` | EMCAD [`lib/decoders.py`](https://github.com/SLDGroup/EMCAD/blob/26c9c31f73/lib/decoders.py) | modified copy (TA decoder ablation) |
| `original_code/2_cascade/unused lib/decoders_3_base-crfb-lgag.py` | EMCAD [`lib/decoders.py`](https://github.com/SLDGroup/EMCAD/blob/26c9c31f73/lib/decoders.py) | modified copy (TA decoder ablation) |
| `original_code/2_cascade/unused lib/decoders_4_base-crfb+sg.py` | EMCAD [`lib/decoders.py`](https://github.com/SLDGroup/EMCAD/blob/26c9c31f73/lib/decoders.py) | modified copy (TA decoder ablation) |
| `original_code/2_cascade/unused lib/decoders_emcad.py` | EMCAD [`lib/decoders.py`](https://github.com/SLDGroup/EMCAD/blob/26c9c31f73/lib/decoders.py) | near-verbatim copy |
| `original_code/2_cascade/unused lib/decoders_cascade.py` | CASCADE [`lib/decoders.py`](https://github.com/SLDGroup/CASCADE) (same UT Austin Research License as EMCAD) | earlier CASCADE decoder |
| `original_code/2_cascade/unused lib/decoders_ffd.py` | EMCAD [`lib/decoders.py`](https://github.com/SLDGroup/EMCAD/blob/26c9c31f73/lib/decoders.py) | modified copy (TA decoder ablation) |
| `original_code/2_cascade/unused lib/decoders_simple.py` | EMCAD [`lib/decoders.py`](https://github.com/SLDGroup/EMCAD/blob/26c9c31f73/lib/decoders.py) | modified copy (TA decoder ablation) |
| `original_code/2_cascade/unused lib/networks1.py` | EMCAD [`lib/networks.py`](https://github.com/SLDGroup/EMCAD/blob/26c9c31f73/lib/networks.py) | wraps the EMCAD decoder (PVT_CASCADE) |
| `original_code/2_cascade/utils/dataloader.py` | EMCAD [`utils/dataloader.py`](https://github.com/SLDGroup/EMCAD/blob/26c9c31f73/utils/dataloader.py) | modified copy |
| `original_code/2_cascade/utils/dataset_ACDC.py` | EMCAD [`utils/dataset_ACDC.py`](https://github.com/SLDGroup/EMCAD/blob/26c9c31f73/utils/dataset_ACDC.py) | near-verbatim copy |
| `original_code/2_cascade/utils/dataset_synapse.py` | EMCAD [`utils/dataset_synapse.py`](https://github.com/SLDGroup/EMCAD/blob/26c9c31f73/utils/dataset_synapse.py) | near-verbatim copy |
| `original_code/2_cascade/utils/format_conversion.py` | EMCAD [`utils/format_conversion.py`](https://github.com/SLDGroup/EMCAD/blob/26c9c31f73/utils/format_conversion.py) | near-verbatim copy |
| `original_code/2_cascade/utils/preprocess_synapse_data.py` | EMCAD [`utils/preprocess_synapse_data.py`](https://github.com/SLDGroup/EMCAD/blob/26c9c31f73/utils/preprocess_synapse_data.py) | near-verbatim copy |
| `original_code/2_cascade/utils/utils.py` | EMCAD [`utils/utils.py`](https://github.com/SLDGroup/EMCAD/blob/26c9c31f73/utils/utils.py) | near-verbatim copy |
| `original_code/3_dino_v3/lib/decoders.py` | EMCAD [`lib/decoders.py`](https://github.com/SLDGroup/EMCAD/blob/26c9c31f73/lib/decoders.py) | near-verbatim copy |
| `original_code/3_dino_v3/lib/pvtv2.py` | Polyp-PVT [`lib/pvtv2.py`](https://github.com/DengPingFan/Polyp-PVT/blob/main/lib/pvtv2.py) | near-verbatim copy |
| `original_code/3_dino_v3/utils/dataloader.py` | EMCAD [`utils/dataloader.py`](https://github.com/SLDGroup/EMCAD/blob/26c9c31f73/utils/dataloader.py) | modified copy |
| `original_code/3_dino_v3/utils/utils.py` | EMCAD [`utils/utils.py`](https://github.com/SLDGroup/EMCAD/blob/26c9c31f73/utils/utils.py) | near-verbatim copy |
| `original_code/4_non_learnable/lib/decoders.py` | EMCAD [`lib/decoders.py`](https://github.com/SLDGroup/EMCAD/blob/26c9c31f73/lib/decoders.py) | near-verbatim copy |
| `original_code/4_non_learnable/lib/networks_pvtv2_cascade.py` | EMCAD [`lib/networks.py`](https://github.com/SLDGroup/EMCAD/blob/26c9c31f73/lib/networks.py) | wraps the EMCAD decoder (PVT_CASCADE) |
| `original_code/4_non_learnable/lib/pvtv2.py` | Polyp-PVT [`lib/pvtv2.py`](https://github.com/DengPingFan/Polyp-PVT/blob/main/lib/pvtv2.py) | near-verbatim copy |
| `original_code/4_non_learnable/utils/dataloader.py` | EMCAD [`utils/dataloader.py`](https://github.com/SLDGroup/EMCAD/blob/26c9c31f73/utils/dataloader.py) | modified copy |
| `original_code/4_non_learnable/utils/utils.py` | EMCAD [`utils/utils.py`](https://github.com/SLDGroup/EMCAD/blob/26c9c31f73/utils/utils.py) | near-verbatim copy |

`original_code/2_cascade/unused lib/vit_seg_*.py` and `cnn_vit_backbone.py` come from
[TransUNet](https://github.com/Beckschen/TransUNet) (Apache-2.0) and are kept with that attribution.
