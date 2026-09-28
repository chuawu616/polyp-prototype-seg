# Trained checkpoints (not in git)

Fifteen representative checkpoints (≈1.7 GB) are kept outside git. Put them here under the names below;
`python tools/test.py <config> checkpoints/<name>.pth` evaluates them. They were produced by the original research
code and are converted automatically on load.

| File | Config | Original run | mDice (unified) | Log-verified |
|---|---|---|---|---|
| `pvt_linear.pth` | `configs/pvt_linear.yaml` | `non_learnable/polyp/baseline_100epoch` | 0.872 | ✓ |
| `pvt_proto_fb88.pth` | `configs/pvt_proto_fb88.yaml` | `cascade/prototype_cascade/baseline_prototype_fb_88` | 0.876 | ✓ |
| `pvt_proto_fb88_rerun.pth` | `configs/pvt_proto_fb88.yaml` | `cascade/prototype_cascade/baseline_prototype_fb_88_v2` | 0.866 | ✗ (no log) |
| `pvt_proto_kl_fb33.pth` | `configs/pvt_proto_kl_fb33.yaml` | `cascade/prototype_cascade/prototype_kl_loss_0.1_fb_33` | 0.870 | ✓ |
| `pvt_specialize_fhb524_warm10.pth` | `configs/pvt_specialize_fhb524_warm10.yaml` | `cascade/prototype_cascade/baseline_prototype_fhb_524_warm_10` | 0.874 | ✓ |
| `pvt_specialize_hybrid_fhb424_warm10.pth` | `configs/pvt_specialize_hybrid_fhb424_warm10.yaml` | `cascade/prototype_cascade/baseline_prototype_hybrid_fhb_424_extract_warm` | 0.870 | ✓ |
| `dinov3_frozen_mlp.pth` | `configs/dinov3_frozen_mlp.yaml` | `dino_v3/polyp/vits16plus/baseline` | 0.785 | ✓ |
| `dinov3_concat_ft_mlp.pth` | `configs/dinov3_concat_ft_mlp.yaml` | `dino_v3/polyp/vits16plus/concat_24681012_fine` | 0.869 | ✓ |
| `dinov3_concat_ft_proto_fb44.pth` | `configs/dinov3_concat_ft_proto_fb44.yaml` | `dino_v3/prototype_cascade/vits16plus/baseline_prototype_fb_44_concat_fine` | 0.870 | ✓ |
| `dinov3_specialize_fhb424_warm10.pth` | `configs/dinov3_specialize_fhb424_warm10.yaml` | `dino_v3/prototype_cascade/vits16plus/baseline_prototype_fhb_424_specialize_warm_10` | 0.869 | ✓ |
| `nonlearnable_v3_dd4.pth` | `configs/nonlearnable_v3_dd4.yaml` | `non_learnable/prototype_cascade/prototypev3_fb55_dd4_ppc_1` | 0.869 | ✓ |
| `nonlearnable_v3_dd4_imagenet_init.pth` | `configs/nonlearnable_v3_dd4.yaml` | `non_learnable/prototype_cascade/prototypev3_fb55_dd4_ppc_1_wo_pretrained` | 0.856 | ✓ |
| `nonlearnable_v0_fb33.pth` | `configs/nonlearnable_v0_fb33.yaml` | `non_learnable/prototype_cascade/prototype_v0_fb33` | 0.845 | ✓ |
| `pseudo_fb33.pth` | `configs/pseudo_fb33.yaml` | `non_learnable/prototype_pseudo/prototype_pseudo_fb33` | 0.857 | ✓ |
| `pseudo_fb33_superpixel.pth` | `configs/pseudo_fb33_superpixel.yaml` | `non_learnable/prototype_pseudo/prototype_pseudo_fb33_sp` | 0.847 | ✓ |

`nonlearnable_v3_dd4_imagenet_init` was trained without the structure-loss-pretrained initialisation
(`train.init_from=null`); whether its KMeans centre initialisation was also removed is not recorded.
The other 55 checkpoints of the project are not distributed; their scores are in
[`docs/results/reevaluation.csv`](../docs/results/reevaluation.csv).
