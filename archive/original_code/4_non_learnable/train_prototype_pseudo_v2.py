import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from lib.networks_prototype_pseudo_v2 import Prototype_Pseudo
from utils.dataloader_pseudo import get_loader, test_dataset
from utils.utils import clip_gradient, AvgMeter


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# ── Loss Functions ─────────────────────────────────────────────────────────────

def ce_loss(pred_multiclass, pseudo_labels, num_prototype, img_size):
    """
    Multi-class CE loss：seg_head（K*M class）vs Sinkhorn pseudo-label。

    Pseudo-label 在 d1 解析度（H/4×W/4），需 nearest 上採樣至原始解析度。
    CE loss 使用 categorical label，nearest 上採樣不影響語義正確性。

    Args:
        pred_multiclass: (B, K*M, H, W) logits（原始解析度）
        pseudo_labels:   (B*H_d1*W_d1,) Sinkhorn subclass label
        num_prototype:   M
        img_size:        原始輸入解析度（int）
    """
    B = pred_multiclass.shape[0]
    h_d1 = img_size // 4
    w_d1 = img_size // 4

    # Nearest 上採樣 pseudo-label 至原始解析度
    pseudo_spatial = pseudo_labels.view(B, 1, h_d1, w_d1).float()
    pseudo_full    = F.interpolate(
        pseudo_spatial, size=(img_size, img_size), mode='nearest'
    ).squeeze(1).long()  # (B, H, W)

    return F.cross_entropy(pred_multiclass, pseudo_full)


def dice_loss(fg_pred, gt_mask):
    """
    Binary Dice loss：foreground probability vs binary GT mask。

    Args:
        fg_pred:  (B, 1, H, W) foreground 機率，由 _binary_from_multiclass 推導
        gt_mask:  (B, 1, H, W) binary ground truth [0, 1]
    """
    inter = (fg_pred * gt_mask).sum(dim=(2, 3))
    union = fg_pred.sum(dim=(2, 3)) + gt_mask.sum(dim=(2, 3))
    return (1 - (2 * inter + 1) / (union + 1)).mean()


# ── Test ───────────────────────────────────────────────────────────────────────

def test(model, path, dataset, img_size):
    data_path  = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root    = '{}/masks/'.format(data_path)
    model.eval()
    num_images  = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, img_size)

    total_dice = 0.0
    for i in range(num_images):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        with torch.no_grad():
            res = model(image)
            if isinstance(res, tuple):
                res = res[0]   # (B, 2, H, W) binary probability

            # FG channel（index 1）
            res = res[:, 1, :, :].unsqueeze(1)
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            res = np.where(res >= 0.5, 1, 0)
            intersection = (res * gt).sum()
            dice = (2. * intersection) / (res.sum() + gt.sum() + 1e-8)
            total_dice += dice

    return total_dice / num_images, num_images


# ── Train ──────────────────────────────────────────────────────────────────────

def train(train_loader, model, optimizer, epoch, opt, trackers):
    model.train()
    global best

    loss_record      = AvgMeter()
    loss_ce_record   = AvgMeter()
    loss_dice_record = AvgMeter()

    total_step = len(train_loader)

    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts, sp_maps = pack
        images  = images.cuda()
        gts     = gts.cuda()                  # (B, 1, H, W)
        sp_maps = sp_maps.cuda()              # (B, H, W) long
        gts_idx = gts.squeeze(1).long()       # (B, H, W)

        # Forward
        out_dict = model(images, gt_semantic_seg=gts_idx, sp_map=sp_maps)

        seg_multiclass = out_dict['seg']          # (B, K*M, H, W) logits
        seg_binary     = out_dict['seg_binary']   # (B, 2, H, W) 機率
        pseudo_labels  = out_dict['pseudo_labels'] # (B*H_d1*W_d1,)
        similarity_map = out_dict['similarity_map']

        loss = torch.tensor(0.0).cuda()

        # CE loss：seg_head 學習 Sinkhorn pseudo-label（multi-class）
        if opt.use_ce_loss:
            loss_ce = ce_loss(seg_multiclass, pseudo_labels, opt.num_prototype, opt.img_size)
            loss = loss + opt.weight_ce * loss_ce
            loss_ce_record.update(loss_ce.data, opt.batchsize)
            trackers['step_ce'].append(loss_ce.item())

        # Dice loss：binary prediction 監督 FG/BG 精度
        if opt.use_dice_loss:
            fg_pred   = seg_binary[:, 1:2, :, :]  # (B, 1, H, W)
            loss_dice = dice_loss(fg_pred, gts)
            loss = loss + opt.weight_dice * loss_dice
            loss_dice_record.update(loss_dice.data, opt.batchsize)
            trackers['step_dice'].append(loss_dice.item())

        if loss.requires_grad:
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            loss_record.update(loss.data, opt.batchsize)
            trackers['step_total'].append(loss.item())

        # ── Tracker：Similarity Map Argmax ─────────────────────────────────
        # similarity_map: (B, K*M, H, W) → argmax → prototype assignment
        pred_proto  = similarity_map.argmax(dim=1).detach()  # (B, H, W)
        B           = images.shape[0]
        pred_flat   = pred_proto.cpu().numpy().reshape(B, -1)   # (B, H*W)
        pred_proto_flat = pred_proto.view(-1)
        M = opt.num_prototype

        # Step-level competition
        fg_mask_step  = (pred_proto_flat >= M) & (pred_proto_flat < 2 * M)
        fg_pixels_step = pred_proto_flat[fg_mask_step]
        fg_counts = (
            torch.bincount(fg_pixels_step - M, minlength=M).cpu().numpy()
            if len(fg_pixels_step) > 0 else np.zeros(M)
        )
        bg_mask_step  = pred_proto_flat < M
        bg_pixels_step = pred_proto_flat[bg_mask_step]
        bg_counts = (
            torch.bincount(bg_pixels_step, minlength=M).cpu().numpy()
            if len(bg_pixels_step) > 0 else np.zeros(M)
        )
        for k in range(M):
            trackers['step_fg_argmax'][k].append(fg_counts[k])
            trackers['step_bg_argmax'][k].append(bg_counts[k])

        # Epoch-level image collapse & prototype ratio
        for b_idx in range(B):
            img = pred_flat[b_idx]

            fg_mask = (img >= M) & (img < 2 * M)
            fg_pix  = img[fg_mask]
            if len(fg_pix) > 0:
                counts    = np.bincount(fg_pix.astype(int) - M, minlength=M)
                max_ratio = counts.max() / len(fg_pix)
                if max_ratio > 0.50: trackers['fg_collapse_50'][epoch] += 1
                if max_ratio > 0.70: trackers['fg_collapse_70'][epoch] += 1
                if max_ratio > 0.90: trackers['fg_collapse_90'][epoch] += 1
                trackers['fg_valid_img_count'][epoch] += 1

            bg_mask = img < M
            bg_pix  = img[bg_mask]
            if len(bg_pix) > 0:
                counts    = np.bincount(bg_pix.astype(int), minlength=M)
                max_ratio = counts.max() / len(bg_pix)
                if max_ratio > 0.50: trackers['bg_collapse_50'][epoch] += 1
                if max_ratio > 0.70: trackers['bg_collapse_70'][epoch] += 1
                if max_ratio > 0.90: trackers['bg_collapse_90'][epoch] += 1
                trackers['bg_valid_img_count'][epoch] += 1

        for k in range(M):
            trackers['proto_counts_bg'][epoch][k] += np.sum(pred_flat == k)
            trackers['proto_counts_fg'][epoch][k] += np.sum(pred_flat == (M + k))

        # ── Log ───────────────────────────────────────────────────────────
        if i % 20 == 0 or i == total_step:
            log_parts = [
                f'Epoch [{epoch:03d}/{opt.epoch:03d}]',
                f'Step [{i:04d}/{total_step:04d}]',
                f'Loss: {loss_record.show():.4f}'
            ]
            if opt.use_ce_loss:   log_parts.append(f'CE: {loss_ce_record.show():.4f}')
            if opt.use_dice_loss: log_parts.append(f'Dice: {loss_dice_record.show():.4f}')
            log_info = ', '.join(log_parts)
            print(log_info)
            logging.info(log_info)

    # ── Checkpoint ────────────────────────────────────────────────────────
    save_path = opt.train_save
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, 'check.pth'))

    if epoch in opt.save_epochs:
        torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch}.pth'))
        logging.info(f'Saved epoch weight: epoch_{epoch}.pth')

    # ── Validation ────────────────────────────────────────────────────────
    total_dataset_dice = 0
    total_dice   = 0
    total_images = 0
    datasets = ['CVC-ClinicDB', 'Kvasir', 'CVC-300', 'CVC-ColonDB', 'ETIS-LaribPolypDB']

    for dataset in datasets:
        dataset_dice, n_images = test(model, opt.test_path, dataset, opt.img_size)
        total_dataset_dice += dataset_dice
        total_dice  += n_images * dataset_dice
        total_images += n_images
        logging.info(f'epoch: {epoch}, dataset: {dataset}, dice: {dataset_dice}')
        print(dataset, ': ', dataset_dice)
        dict_plot[dataset].append(dataset_dice)

    meandice = total_dice / total_images
    total_dataset_dice /= len(datasets)
    dict_plot['test'].append(meandice)
    print(f'mdice: {total_dataset_dice:.4f}')
    print(f'Validation dice score: {meandice}')
    logging.info(f'Validation dice score: {meandice}')

    if total_dataset_dice > best:
        print(f'### Dice improved {best:.4f} → {total_dataset_dice:.4f}')
        logging.info(f'### Dice improved {best:.4f} → {total_dataset_dice:.4f}')
        best = total_dataset_dice
        torch.save(model.state_dict(), os.path.join(save_path, 'best.pth'))


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    dict_plot = {
        'CVC-300': [], 'CVC-ClinicDB': [], 'Kvasir': [],
        'CVC-ColonDB': [], 'ETIS-LaribPolypDB': [], 'test': []
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',         type=int,   default=30)
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--batchsize',     type=int,   default=16)
    parser.add_argument('--img_size',      type=int,   default=352)
    parser.add_argument('--clip',          type=float, default=0.5)
    parser.add_argument('--decay_rate',    type=float, default=0.1)
    parser.add_argument('--decay_epoch',   type=int,   default=100)
    parser.add_argument('--train_path',    type=str,
                        default='/home/U116med/data/polyp/TrainDataset/')
    parser.add_argument('--test_path',     type=str,
                        default='/home/U116med/data/polyp/TestDataset/')
    parser.add_argument('--train_save',    type=str,
                        default='models/prototype_pseudo/')
    parser.add_argument('--augmentation',  default='False')
    parser.add_argument('--device',        type=int,   default=0)
    parser.add_argument('--name',          type=str,   default='prototype_pseudo_fb33_sp')
    parser.add_argument('--num_prototype', type=int,   default=3,
                        help='每個 class 的 prototype 數量（FG 和 BG 相同）')
    parser.add_argument('--save_epochs',   type=int, nargs='+',
                        default=[1, 2, 3, 4, 5, 10, 20, 50, 100])

    # Loss flags
    parser.add_argument('--use_ce_loss',   type=str2bool, default=True,
                        help='CE loss：seg_head vs Sinkhorn pseudo-label（multi-class）')
    parser.add_argument('--use_dice_loss', type=str2bool, default=True,
                        help='Dice loss：binary prediction vs binary GT')
    parser.add_argument('--weight_ce',     type=float, default=1.0)
    parser.add_argument('--weight_dice',   type=float, default=1.0)

    # Superpixel
    parser.add_argument('--sp_root',       type=str,   default='/home/U116med/wch_code/non_learnable/superpixels/',
                        help='GT-guided superpixel .npy 目錄（None = pixel-level Sinkhorn）'
                             '  e.g. /home/.../TrainDataset/superpixels_gt/')

    opt = parser.parse_args()

    logging.basicConfig(
        filename=f'train_log_{opt.name}.log',
        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p'
    )

    opt.train_save = os.path.join(opt.train_save, opt.name)
    os.makedirs(opt.train_save, exist_ok=True)
    torch.cuda.set_device(opt.device)

    model = Prototype_Pseudo(
        num_classes=2,
        num_prototype=opt.num_prototype
    ).cuda()

    best = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    print(optimizer)

    image_root   = '{}/images/'.format(opt.train_path)
    gt_root      = '{}/masks/'.format(opt.train_path)
    train_loader = get_loader(
        image_root, gt_root,
        batchsize=opt.batchsize, trainsize=opt.img_size,
        augmentation=opt.augmentation,
        sp_root=opt.sp_root
    )

    # ── Trackers ──────────────────────────────────────────────────────────
    E = range(1, opt.epoch + 1)
    M = opt.num_prototype
    trackers = {
        # Step-level losses
        'step_ce':    [],
        'step_dice':  [],
        'step_total': [],

        # Step-level prototype competition（similarity_map argmax）
        'step_fg_argmax': {k: [] for k in range(M)},
        'step_bg_argmax': {k: [] for k in range(M)},

        # Epoch-level prototype pixel ratio
        'proto_counts_fg': {e: np.zeros(M) for e in E},
        'proto_counts_bg': {e: np.zeros(M) for e in E},

        # Epoch-level image-level collapse
        'fg_collapse_90':     {e: 0 for e in E},
        'fg_collapse_70':     {e: 0 for e in E},
        'fg_collapse_50':     {e: 0 for e in E},
        'bg_collapse_90':     {e: 0 for e in E},
        'bg_collapse_70':     {e: 0 for e in E},
        'bg_collapse_50':     {e: 0 for e in E},
        'fg_valid_img_count': {e: 0 for e in E},
        'bg_valid_img_count': {e: 0 for e in E},
    }

    print("######## Start Training Prototype Pseudo ########")
    for epoch in range(1, opt.epoch + 1):
        train(train_loader, model, optimizer, epoch, opt, trackers)

    # ── Plots ─────────────────────────────────────────────────────────────
    print("######## Generating Plots... ########")
    epochs_list = list(range(1, opt.epoch + 1))
    save_path   = opt.train_save

    # 1. Loss Curve
    plt.figure(figsize=(12, 6))
    if opt.use_ce_loss and trackers['step_ce']:
        plt.plot(trackers['step_ce'],    label='CE Loss',   alpha=0.8)
    if opt.use_dice_loss and trackers['step_dice']:
        plt.plot(trackers['step_dice'],  label='Dice Loss', alpha=0.8)
    if trackers['step_total']:
        plt.plot(trackers['step_total'], label='Total Loss', alpha=0.6, linestyle='--')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss Value')
    plt.title('Step-wise Loss Curve (Prototype Pseudo)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'loss_curve.png'))
    plt.close()

    # 2. Prototype Pixel Ratio（similarity_map argmax）
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for k in range(M):
        fg_ratios, bg_ratios = [], []
        for e in epochs_list:
            total_fg = np.sum(trackers['proto_counts_fg'][e])
            total_bg = np.sum(trackers['proto_counts_bg'][e])
            fg_ratios.append(
                trackers['proto_counts_fg'][e][k] / total_fg if total_fg > 0 else 0
            )
            bg_ratios.append(
                trackers['proto_counts_bg'][e][k] / total_bg if total_bg > 0 else 0
            )
        axes[0].plot(epochs_list, fg_ratios, label=f'FG_{k}', marker='.')
        axes[1].plot(epochs_list, bg_ratios, label=f'BG_{k}', marker='.')

    for ax, title in zip(axes, ['Foreground Prototype Ratio', 'Background Prototype Ratio']):
        ax.set_title(f'{title} (similarity_map argmax)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Ratio')
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'prototype_ratio_curve.png'))
    plt.close()

    # 3. Image-Level Collapse
    def safe_ratio(num_dict, den_dict, e):
        return num_dict[e] / den_dict[e] if den_dict[e] > 0 else 0

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, class_name, c90, c70, c50, cnt, colors in [
        (axes[0], 'Foreground',
         trackers['fg_collapse_90'], trackers['fg_collapse_70'], trackers['fg_collapse_50'],
         trackers['fg_valid_img_count'], ('darkred', 'red', 'salmon')),
        (axes[1], 'Background',
         trackers['bg_collapse_90'], trackers['bg_collapse_70'], trackers['bg_collapse_50'],
         trackers['bg_valid_img_count'], ('darkblue', 'blue', 'cornflowerblue')),
    ]:
        ax.plot(epochs_list, [safe_ratio(c90, cnt, e) for e in epochs_list],
                label='>90% Dominance', marker='o', color=colors[0])
        ax.plot(epochs_list, [safe_ratio(c70, cnt, e) for e in epochs_list],
                label='>70% Dominance', marker='s', color=colors[1], alpha=0.7)
        ax.plot(epochs_list, [safe_ratio(c50, cnt, e) for e in epochs_list],
                label='>50% Dominance', marker='^', color=colors[2], alpha=0.7)
        ax.set_title(f'{class_name} Image-Level Collapse')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Ratio of Images')
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'image_level_collapse_curve.png'))
    plt.close()

    # 4. Step-wise Prototype Competition
    def moving_average(a, n=50):
        if len(a) < n:
            return a
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, class_name, argmax_key in [
        (axes[0], 'Foreground', 'step_fg_argmax'),
        (axes[1], 'Background', 'step_bg_argmax'),
    ]:
        counts = np.array([trackers[argmax_key][k] for k in range(M)])
        totals = counts.sum(axis=0) + 1e-8
        ratios = counts / totals
        for k in range(M):
            smoothed = moving_average(ratios[k])
            label = f'{"FG" if "fg" in argmax_key else "BG"}_{k}'
            ax.plot(range(len(smoothed)), smoothed, label=label, alpha=0.8)
        ax.set_title(f'Step-wise {class_name} Prototype Competition')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Pixel Selection Ratio')
        ax.set_ylim([-0.05, 1.05])
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'step_competition_curve.png'))
    plt.close()

    print(f"Plots saved to {save_path}")