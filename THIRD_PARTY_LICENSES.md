# Third-party components

The code written for this repository is MIT-licensed ([LICENSE](LICENSE)). It builds on the following work,
which keeps its own license.

| Component | Where it is used | Origin | License | In this repository |
|---|---|---|---|---|
| PVTv2 backbone | `protoseg_polyp/models/pvtv2.py` | [whai362/PVT](https://github.com/whai362/PVT) (branch v2, `detection/pvt_v2.py`) | Apache-2.0 ([text](licenses/PVT-Apache-2.0.txt)) | included, modified (mmdet/mmcv dependencies removed; see file header) |
| EMCAD decoder | `protoseg_polyp/models/third_party/emcad_decoders.py` | [SLDGroup/EMCAD](https://github.com/SLDGroup/EMCAD) (commit `26c9c31f73`, `lib/decoders.py`) | UT Austin Research License — academic / research / personal use, **no redistribution** | **not included**; downloaded unmodified by `tools/fetch_third_party.py`. Using this repository with the EMCAD decoder is subject to that license (non-commercial). |
| Sinkhorn-Knopp assignment, EMA update, PPC / PPD losses | `protoseg_polyp/ops.py`, `engine/losses.py` | [tfzhou/ProtoSeg](https://github.com/tfzhou/ProtoSeg) | MIT ([text](licenses/ProtoSeg-MIT.txt)) | re-implemented / adapted |
| DINOv3 backbones | `protoseg_polyp/models/encoders.py` | [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | DINOv3 License | **not included**; code fetched by `torch.hub`, weights must be requested from Meta |
| Structure loss (weighted BCE + IoU), multi-scale training and test-time Dice | `engine/losses.py`, `engine/trainer.py`, `engine/evaluate.py` | F3Net / PraNet / Polyp-PVT polyp-segmentation code | — | re-implemented from the published method |
| SSL-ALPNet (Stage 0 only) | `archive/0_ssl_fss/` | [cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation) | MIT ([text](licenses/SSL-ALPNet-MIT.txt)) | author's additions + patches against upstream |
| TransUNet ViT configs (Stage 2 decoder experiments of the TA) | `archive/original_code/2_cascade/unused lib/vit_seg_*.py`, `cnn_vit_backbone.py` | [Beckschen/TransUNet](https://github.com/Beckschen/TransUNet) | Apache-2.0 | archived, not used by the package |
| PraNet polyp benchmark split (Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB, CVC-300, ETIS-LaribPolypDB) | data | [PraNet](https://github.com/DengPingFan/PraNet) | dataset licenses of the original providers | **not included** |

Trained checkpoints are not distributed in this repository (models with the PVTv2 + EMCAD encoder contain trained
EMCAD decoder weights).
