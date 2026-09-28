# Pre-trained backbones (not in git)

| File | Source |
|---|---|
| `pvt_v2_b2.pth` | ImageNet PVTv2-b2 from the [PVT repository](https://github.com/whai362/PVT/releases/tag/v2) (`pvt_v2_b2.pth`) |
| `dinov3_vits16plus_pretrain.pth` (and `vitb16`, `vitl16`) | [DINOv3](https://github.com/facebookresearch/dinov3) — requires accepting Meta's license; rename the downloaded file to this name |

DINOv3 model code is fetched by `torch.hub` from `facebookresearch/dinov3` on first use
(set `model.encoder.hub_repo=<local clone> model.encoder.hub_source=local` to work offline).
