# 檔案: train_stage3_fixed.py

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from models.MetricSegmenter import MetricSegmenter
from dataloaders.FinalTrainingDataset import FinalTrainingDataset
from dataloaders.SupervisedPolypDataset import SupervisedPolypDataset as ValidationDataset
from util.losses import CombinedSegLoss, PrototypeMetricLoss
from util.metric import SegmentationMetric


# -------------------------------------------------------------
# 1. Warmup + Cosine decay Scheduler（非常標準）
# 注意：LambdaLR 的 lr_lambda 必須返回純粹的 Python float（not Tensor）
# -------------------------------------------------------------
def build_warmup_cosine_scheduler(optimizer, warmup_epochs, max_epochs):
    """
    epoch-based warmup+cosine scheduler.
    warmup_epochs, max_epochs are in epochs (integers).
    lr_lambda receives current epoch (int, 0-based) and must return float multiplier.
    """
    def lr_lambda(current_epoch: int):
        # linear warmup (epoch granularity)
        if current_epoch < 0:
            return 1.0
        if warmup_epochs <= 0:
            # no warmup: pure cosine over max_epochs
            progress = float(min(current_epoch, max_epochs)) / float(max(1, max_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        # cosine decay after warmup
        denom = max(1, (max_epochs - warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(denom)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# -------------------------------------------------------------
# 2. 評估函數（更穩健：用 softmax + threshold 而不是 argmax）
# -------------------------------------------------------------
def evaluate(model, dataloader, device, threshold=0.5):
    model.eval()
    metric = SegmentationMetric(num_classes=2)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="驗證中", leave=False):
            images, masks = batch[0].to(device), batch[1].to(device)
            logits, _ = model(images)

            # 使用 softmax -> 取前景機率，再 threshold 成 0/1 mask
            probs = torch.softmax(logits, dim=1)[:, 1]  # shape [B, H, W]
            preds = (probs > threshold).long()

            # 確保輸入 metric 的為 cpu numpy / long
            metric.update(masks.cpu(), preds.cpu())

    scores = metric.get_scores()
    return scores["Class_Dice"][1]


# -------------------------------------------------------------
# 3. 主訓練流程
# -------------------------------------------------------------
def train_stage3(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------
    # Data
    # ------------------------------
    train_dataset = FinalTrainingDataset(
        image_dir=cfg["train_image_dir"],
        binary_mask_dir=cfg["train_mask_dir"],
        sublabel_dir=cfg["sublabel_dir"],
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)

    val_dataset = ValidationDataset(
        image_dir=cfg["val_image_dir"],
        mask_dir=cfg["val_mask_dir"],
        return_original=False
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg["val_batch_size"], shuffle=False, num_workers=4)

    # ------------------------------
    # Model
    # ------------------------------
    model = MetricSegmenter(cfg=cfg["model_cfg"]).to(device)

    # ------------------------------
    # Loss
    # ------------------------------
    seg_loss_fn = CombinedSegLoss()
    metric_loss_fn = PrototypeMetricLoss(n_fg_clusters=cfg["model_cfg"]["num_fg_prototypes"])
    metric_weight = cfg["metric_loss_weight"]

    # ------------------------------
    # Optimizer (分層 LR)
    # ------------------------------
    param_groups = [
        {"params": model.encoder.parameters(), "lr": cfg["encoder_lr"]},
        {"params": model.decoder.parameters(), "lr": cfg["decoder_lr"]},
        {"params": model.segmentation_head.parameters(), "lr": cfg["head_lr"]},
        {"params": model.fg_prototypes, "lr": cfg["prototype_lr"]},
    ]

    optimizer = AdamW(param_groups, weight_decay=cfg["weight_decay"])

    # ------------------------------
    # Warmup + Cosine Decay (epoch-based)
    # ------------------------------
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=cfg.get("warmup_epochs", 0),
        max_epochs=cfg["epochs"],
    )

    # ------------------------------
    # Mixed Precision
    # ------------------------------
    scaler = GradScaler()

    # ------------------------------
    # Training
    # ------------------------------
    best_val = -1

    for epoch in range(cfg["epochs"]):

        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")

        for images, masks, sublabels in pbar:

            images = images.to(device)
            masks = masks.to(device)
            sublabels = sublabels.to(device)

            optimizer.zero_grad()

            # AMP 前向
            with autocast():
                logits, fg_sim = model(images)

                loss_seg = seg_loss_fn(logits, masks)

                # match resolution
                sublabel_small = F.interpolate(
                    sublabels.unsqueeze(1).float(),
                    size=fg_sim.shape[-2:], mode="nearest"
                ).squeeze(1).long()

                loss_metric = metric_loss_fn(fg_sim, sublabel_small)

                loss = loss_seg + metric_weight * loss_metric

            # AMP 反向 + 梯度縮放
            scaler.scale(loss).backward()

            # 防止梯度爆炸
            scaler.unscale_(optimizer)  # ensure grads are unscaled before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # 更新 LR (epoch-level)
        scheduler.step()

        # ------------------------------
        # Eval
        # ------------------------------
        val_dice = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch+1}] Train Loss={total_loss/len(train_loader):.4f} | Val Dice={val_dice:.4f}")

        # ------------------------------
        # Save Best
        # ------------------------------
        if val_dice > best_val:
            best_val = val_dice
            os.makedirs(cfg["output_dir"], exist_ok=True)
            save_path = os.path.join(cfg["output_dir"], "best_stage3.pth")
            torch.save(model.state_dict(), save_path)
            print(f"★★ 新最佳模型 ★★ Dice={best_val:.4f}  → 已保存至 {save_path}")


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
if __name__ == "__main__":

    cfg = {
        "train_image_dir": "/home/U116med/data/polyp/TrainDataset/images/",
        "train_mask_dir": "/home/U116med/data/polyp/TrainDataset/masks/",
        "sublabel_dir": "./polypdata/sublabels_consistent_k8/",

        "val_image_dir": "/home/U116med/data/polyp/TestDataset/test/images/",
        "val_mask_dir": "/home/U116med/data/polyp/TestDataset/test/masks/",

        "batch_size": 4,
        "val_batch_size": 4,
        "epochs": 100,
        "warmup_epochs": 5,          # ★ warmup (in epochs)

        "model_cfg": {
            "stage2_pretrained_path": "./pretrained_stage2_weighted/stage2_epoch_20.pth",
            "decoder_out_channels": 256,
            "num_fg_prototypes": 4,
        },

        "encoder_lr": 1e-6,
        "decoder_lr": 1e-5,
        "head_lr": 1e-4,
        "prototype_lr": 1e-4,
        "weight_decay": 1e-2,

        "metric_loss_weight": 0.05,

        "output_dir": "./final_model_metric_revised/",
    }

    train_stage3(cfg)
