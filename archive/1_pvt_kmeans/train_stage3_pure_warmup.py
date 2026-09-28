# 檔案: train_stage3_pure_fixed.py

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

from models.PrototypeSegmenter_Pure import PrototypeSegmenterPure
from dataloaders.FinalTrainingDataset import FinalTrainingDataset
from dataloaders.SupervisedPolypDataset import SupervisedPolypDataset as ValidationDataset
from util.losses_reg import StructureLoss, SublabelLoss, RegularizationLoss
from util.metric import SegmentationMetric


# -----------------------------
# Scheduler helper: warmup (steps) + cosine decay (steps)
# -----------------------------
def build_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps):
    """
    Returns a LambdaLR where lr scales:
      - linearly from 0 -> 1 over warmup_steps
      - then cosine decay from 1 -> 0 over (total_steps - warmup_steps)
    The lambda receives current_step (0-based), as expected by LambdaLR.
    """
    def lr_lambda(current_step: int):
        if current_step < 0:
            return 1.0
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


# -----------------------------
# Evaluate
# -----------------------------
def evaluate(model, dataloader, device):
    model.eval()
    metric = SegmentationMetric(num_classes=2)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="驗證中...", leave=False):
            # dataloader yields (images, masks, maybe extras) - only take first two
            if isinstance(batch, (list, tuple)):
                images, true_binary_masks = batch[0], batch[1]
            else:
                # defensive fallback
                images = batch
                true_binary_masks = None

            images = images.to(device)
            true_binary_masks = true_binary_masks.to(device)

            logits_binary, _ = model(images)
            pred_binary = logits_binary.argmax(dim=1)
            metric.update(true_binary_masks.cpu(), pred_binary.cpu())

    scores = metric.get_scores()
    return scores["Class_Dice"][1]


# -----------------------------
# Main training
# -----------------------------
def train_stage3_pure(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Data
    # -------------------------
    train_dataset = FinalTrainingDataset(
        image_dir=cfg['train_image_dir'],
        binary_mask_dir=cfg['train_mask_dir'],
        sublabel_dir=cfg['sublabel_dir']
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=cfg.get('num_workers', 4), pin_memory=True)

    val_dataset = ValidationDataset(image_dir=cfg['val_image_dir'], mask_dir=cfg['val_mask_dir'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['val_batch_size'], shuffle=False,
                            num_workers=cfg.get('num_workers', 4), pin_memory=True)

    # -------------------------
    # Model
    # -------------------------
    model = PrototypeSegmenterPure(cfg=cfg['model_cfg']).to(device)

    # optionally load pretrained weights for backbone or whole model if provided
    if cfg.get('stage2_pretrained_load', False):
        path = cfg['model_cfg'].get('stage2_pretrained_path', None)
        if path and os.path.isfile(path):
            ckpt = torch.load(path, map_location='cpu')
            # assume it's a state_dict for the model
            try:
                model.load_state_dict(ckpt, strict=False)
                print(f"[INFO] Loaded pretrained weights from {path}")
            except Exception as e:
                print(f"[WARN] Failed to strictly load pretrained weights: {e}. Attempting partial load.")
                # partial load
                model_state = model.state_dict()
                pretrained_items = {k: v for k, v in ckpt.items() if k in model_state and v.shape == model_state[k].shape}
                model_state.update(pretrained_items)
                model.load_state_dict(model_state)
                print(f"[INFO] Partial load done ({len(pretrained_items)} tensors matched).")

    # -------------------------
    # Loss
    # -------------------------
    structure_loss_fn = StructureLoss()
    sublabel_loss_fn = SublabelLoss()
    aux_weight = cfg.get('aux_loss_weight', 0.3)
    reg_loss_fn = RegularizationLoss(w_scale=0.5, w_iron=0.01) 

    # -------------------------
    # Optimizer (param groups)
    # -------------------------
    # default safer weight decay if not in config
    weight_decay = cfg.get('weight_decay', 1e-4)

    param_groups = [
        {'params': model.encoder.parameters(), 'lr': cfg.get('encoder_lr', 1e-5)},
        {'params': model.decoder.parameters(), 'lr': cfg.get('decoder_lr', 1e-4)},
        {'params': getattr(model, 'prototypes', []), 'lr': cfg.get('prototype_lr', 1e-3)}
    ]
    optimizer = AdamW(param_groups, weight_decay=weight_decay)

    # -------------------------
    # Scheduler: compute total_steps & warmup_steps (iteration-based)
    # -------------------------
    epochs = cfg['epochs']
    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch

    # warmup can be specified either as steps or ratio
    if 'warmup_steps' in cfg:
        warmup_steps = int(cfg['warmup_steps'])
    else:
        warmup_ratio = cfg.get('warmup_ratio', 0.01)  # default 1% of total steps
        warmup_steps = int(total_steps * warmup_ratio)

    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    # -------------------------
    # AMP scaler and gradient clip
    # -------------------------
    scaler = GradScaler()
    clip_norm = cfg.get('clip_grad_norm', 1.0)

    # -------------------------
    # Optionally freeze encoder for first N epochs (useful to protect pretrained features)
    # -------------------------
    freeze_encoder_epochs = cfg.get('freeze_encoder_epochs', 0)
    if freeze_encoder_epochs > 0:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print(f"[INFO] Encoder frozen for first {freeze_encoder_epochs} epochs.")

    # -------------------------
    # Training loop (iteration-based scheduler.step)
    # -------------------------
    print("###### 開始第三階段 (純原型 + Structure Loss) 訓練 ######")
    best_val_dice = -1.0
    global_step = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        # unfreeze if freeze period done
        if epoch == freeze_encoder_epochs and freeze_encoder_epochs > 0:
            for p in model.encoder.parameters():
                p.requires_grad = True
            print(f"[INFO] Encoder unfrozen at epoch {epoch+1}.")

        if epoch < 5:
            ramp_weight = 0.0
        else:
            ramp_weight = min(1.0, (epoch - 5) / 10.0)

        for batch in pbar:
            # support datasets that return more fields; only use first three
            if isinstance(batch, (list, tuple)):
                images = batch[0]
                binary_masks = batch[1]
                sublabel_masks = batch[2] if len(batch) > 2 else None
            else:
                raise RuntimeError("Unexpected batch type from DataLoader")

            images = images.to(device, non_blocking=True)
            binary_masks = binary_masks.to(device, non_blocking=True)
            if sublabel_masks is not None:
                sublabel_masks = sublabel_masks.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast():
                logits_binary, logits_k_class = model(images)

                loss_main = structure_loss_fn(logits_binary, binary_masks)

                # ensure aux target resolution matches logits_k_class
                if sublabel_masks is not None:
                    # if logits_k_class already matches input size then no resize; otherwise resize
                    target_size = logits_k_class.shape[-2:]
                    if sublabel_masks.shape[-2:] != target_size:
                        sublabel_small = F.interpolate(sublabel_masks.unsqueeze(1).float(), size=target_size,
                                                       mode='nearest').squeeze(1).long()
                    else:
                        sublabel_small = sublabel_masks.long()

                    loss_aux = sublabel_loss_fn(logits_k_class, sublabel_small)
                else:
                    loss_aux = torch.tensor(0.0, device=device)
                    
                loss_reg = reg_loss_fn(logits_k_class)
                total_loss = loss_main + aux_weight * loss_aux + ramp_weight * loss_reg


            # backward + step with AMP
            scaler.scale(total_loss).backward()

            # clip grads (safety)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            scaler.step(optimizer)
            scaler.update()

            # scheduler.step per iteration
            scheduler.step()
            global_step += 1

            epoch_loss += total_loss.item()
            pbar.set_postfix(loss=f"{total_loss.item():.4f}", main=f"{loss_main.item():.4f}", aux=f"{loss_aux.item():.4f}")

        # -------------------------
        # End-of-epoch eval & save
        # -------------------------
        val_dice = evaluate(model, val_loader, device)
        avg_loss = epoch_loss / float(steps_per_epoch)
        print(f"[Epoch {epoch+1}/{epochs}] TrainLoss={avg_loss:.4f} ValDice={val_dice:.4f} (step {global_step}/{total_steps})")

        # save best
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            os.makedirs(cfg['output_dir'], exist_ok=True)
            save_path = os.path.join(cfg['output_dir'], "best_model_pure_fixed.pth")
            # save model and optimizer+scaler+scheduler metadata (helpful to continue training)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'scheduler_last_epoch': scheduler.last_epoch,
                'global_step': global_step,
                'best_val_dice': best_val_dice,
                'cfg': cfg
            }, save_path)
            print(f"****** 新最佳模型 (Dice {best_val_dice:.4f}) 已保存到: {save_path} ******")

    print("訓練結束。最佳 Val Dice:", best_val_dice)


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == '__main__':
    config = {
        'train_image_dir': '/home/U116med/data/polyp/TrainDataset/images/',
        'train_mask_dir': '/home/U116med/data/polyp/TrainDataset/masks/',
        'sublabel_dir': './polypdata/sublabels_consistent_k8/',
        'val_image_dir': '/home/U116med/data/polyp/TestDataset/test/images/',
        'val_mask_dir': '/home/U116med/data/polyp/TestDataset/test/masks/',

        'batch_size': 4,
        'val_batch_size': 4,
        'epochs': 100,

        'model_cfg': {
            'stage2_pretrained_path': None,
            'decoder_out_channels': 256,
            'num_fg_prototypes': 4,
            'num_bg_prototypes': 4,
        },

        # learning rates (you can tune)
        'encoder_lr': 1e-5,
        'decoder_lr': 1e-4,
        'prototype_lr': 1e-3,

        # safer default weight decay for fine-tuning; override if you want 1e-2
        'weight_decay': 1e-4,

        # warmup: use warmup_ratio or warmup_steps
        'warmup_ratio': 0.01,   # 1% of total steps by default
        # 'warmup_steps': 200,  # alternatively explicit

        # other training hyperparams
        'aux_loss_weight': 0.3,
        'clip_grad_norm': 1.0,
        'freeze_encoder_epochs': 0,  # optionally freeze encoder for initial epochs
        'num_workers': 4,
        'output_dir': './final_model_pure_reg_sub_wo_pre/',
        # set True if you want to try loading pretrained checkpoint into model
        'stage2_pretrained_load': False,
    }

    train_stage3_pure(config)
