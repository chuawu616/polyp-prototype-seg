import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from lib.networks_prototype_v0 import Prototype_CASCADE_v0
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class PPC(nn.Module):
    """Pixel-Prototype Contrast loss（Cross-Entropy over all K*M prototypes）"""
    def __init__(self, ignore_label=-1):
        super(PPC, self).__init__()
        self.ignore_label = ignore_label

    def forward(self, contrast_logits, contrast_target):
        return F.cross_entropy(
            contrast_logits, contrast_target.long(), ignore_index=self.ignore_label
        )


class PPD(nn.Module):
    """Pixel-Prototype Distance loss（push assigned prototype similarity → 1）"""
    def __init__(self, ignore_label=-1):
        super(PPD, self).__init__()
        self.ignore_label = ignore_label

    def forward(self, contrast_logits, contrast_target):
        valid_mask = (contrast_target != self.ignore_label)
        contrast_logits = contrast_logits[valid_mask, :]
        contrast_target = contrast_target[valid_mask]

        if contrast_logits.shape[0] == 0:
            return torch.tensor(0.0).to(contrast_logits.device)

        logits = torch.gather(contrast_logits, 1, contrast_target[:, None].long())
        return (1 - logits).pow(2).mean()


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def test(model, path, dataset, img_size):
    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num_images = len(os.listdir(gt_root))
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
                res = res[0]
            elif isinstance(res, dict):
                res = res['seg']

            res = res[:, 1, :, :].unsqueeze(1)
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            res = np.where(res >= 0.5, 1, 0)
            intersection = (res * gt).sum()
            dice = (2. * intersection) / (res.sum() + gt.sum() + 1e-8)
            total_dice += dice

    return total_dice / num_images, num_images


def train(train_loader, model, optimizer, epoch, opt, trackers):
    model.train()
    global best

    loss_record = AvgMeter()
    loss_ce_record = AvgMeter()
    loss_struct_record = AvgMeter()
    loss_ppc_record = AvgMeter()
    loss_ppd_record = AvgMeter()

    total_step = len(train_loader)

    criterion_ppc = PPC().cuda()
    criterion_ppd = PPD().cuda()
    criterion_ce = nn.CrossEntropyLoss(ignore_index=-1).cuda()

    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts = pack
        images = images.cuda()
        gts = gts.cuda()
        gts_idx = gts.squeeze(1).long()

        # v0 output: {'seg', 'logits', 'labels', 'similarity_map'}
        out_dict = model(images, gt_semantic_seg=gts_idx)
        seg_logits   = out_dict['seg']            # (B, K, H, W)
        logits       = out_dict['logits']         # (B*H*W, K*M)
        labels       = out_dict['labels']         # (B*H*W,) Sinkhorn subclass label
        similarity_map = out_dict['similarity_map']

        loss = torch.tensor(0.0).cuda()

        if opt.use_ce_loss:
            loss_ce = criterion_ce(seg_logits, gts_idx)
            loss = loss + loss_ce
            loss_ce_record.update(loss_ce.data, opt.batchsize)
            trackers['step_ce'].append(loss_ce.item())

        if opt.use_structure_loss:
            fg_logits = (
                (seg_logits[:, 1, :, :] - seg_logits[:, 0, :, :]) * 10.0
            ).unsqueeze(1)
            loss_struct = structure_loss(fg_logits, gts)
            loss = loss + loss_struct
            loss_struct_record.update(loss_struct.data, opt.batchsize)
            trackers['step_struct'].append(loss_struct.item())

        # PPC & PPD 共用同一組 logits/labels（d1 層 Sinkhorn assignment）
        if opt.use_ppc_loss:
            loss_ppc = criterion_ppc(logits, labels)
            loss = loss + opt.weight_ppc * loss_ppc
            loss_ppc_record.update(loss_ppc.data, opt.batchsize)
            trackers['step_ppc'].append(loss_ppc.item())

        if opt.use_ppd_loss:
            loss_ppd = criterion_ppd(logits, labels)
            loss = loss + opt.weight_ppd * loss_ppd
            loss_ppd_record.update(loss_ppd.data, opt.batchsize)
            trackers['step_ppd'].append(loss_ppd.item())

        if loss.requires_grad:
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            loss_record.update(loss.data, opt.batchsize)
            trackers['step_total'].append(loss.item())

        # ── D1 層統計（similarity_map argmax）──────────────────────────────
        # similarity_map shape: (B, K*M, H, W)
        # argmax → prototype index for each pixel
        pred_proto = similarity_map.argmax(dim=1).detach()  # (B, H, W)
        B = images.shape[0]
        pred_flat = pred_proto.cpu().numpy().reshape(B, -1)  # (B, H*W)

        # Step-level 競爭追蹤
        pred_proto_flat = pred_proto.view(-1)

        fg_mask_step = (pred_proto_flat >= opt.num_prototype) & \
                       (pred_proto_flat < 2 * opt.num_prototype)
        fg_pixels_step = pred_proto_flat[fg_mask_step]
        if len(fg_pixels_step) > 0:
            fg_counts = torch.bincount(
                fg_pixels_step - opt.num_prototype, minlength=opt.num_prototype
            ).cpu().numpy()
        else:
            fg_counts = np.zeros(opt.num_prototype)
        for k in range(opt.num_prototype):
            trackers['step_fg_argmax'][k].append(fg_counts[k])

        bg_mask_step = pred_proto_flat < opt.num_prototype
        bg_pixels_step = pred_proto_flat[bg_mask_step]
        if len(bg_pixels_step) > 0:
            bg_counts = torch.bincount(
                bg_pixels_step, minlength=opt.num_prototype
            ).cpu().numpy()
        else:
            bg_counts = np.zeros(opt.num_prototype)
        for k in range(opt.num_prototype):
            trackers['step_bg_argmax'][k].append(bg_counts[k])

        # Epoch-level image collapse & global ratio
        for b_idx in range(B):
            img = pred_flat[b_idx]

            # Foreground collapse
            fg_mask = (img >= opt.num_prototype) & (img < 2 * opt.num_prototype)
            fg_pixels = img[fg_mask]
            if len(fg_pixels) > 0:
                counts = np.bincount(
                    fg_pixels.astype(int) - opt.num_prototype, minlength=opt.num_prototype
                )
                max_ratio = counts.max() / len(fg_pixels)
                if max_ratio > 0.50: trackers['fg_collapse_50'][epoch] += 1
                if max_ratio > 0.70: trackers['fg_collapse_70'][epoch] += 1
                if max_ratio > 0.90: trackers['fg_collapse_90'][epoch] += 1
                trackers['fg_valid_img_count'][epoch] += 1

            # Background collapse
            bg_mask = img < opt.num_prototype
            bg_pixels = img[bg_mask]
            if len(bg_pixels) > 0:
                counts = np.bincount(
                    bg_pixels.astype(int), minlength=opt.num_prototype
                )
                max_ratio = counts.max() / len(bg_pixels)
                if max_ratio > 0.50: trackers['bg_collapse_50'][epoch] += 1
                if max_ratio > 0.70: trackers['bg_collapse_70'][epoch] += 1
                if max_ratio > 0.90: trackers['bg_collapse_90'][epoch] += 1
                trackers['bg_valid_img_count'][epoch] += 1

        # Global prototype pixel count（用於 ratio curve）
        for k in range(opt.num_prototype):
            trackers['proto_counts_bg'][epoch][k] += np.sum(pred_flat == k)
            trackers['proto_counts_fg'][epoch][k] += np.sum(
                pred_flat == (opt.num_prototype + k)
            )

        # ── Log ───────────────────────────────────────────────────────────
        if i % 20 == 0 or i == total_step:
            log_parts = [
                f'Epoch [{epoch:03d}/{opt.epoch:03d}]',
                f'Step [{i:04d}/{total_step:04d}]',
                f'Loss: {loss_record.show():.4f}'
            ]
            if opt.use_ce_loss:        log_parts.append(f'CE: {loss_ce_record.show():.4f}')
            if opt.use_structure_loss: log_parts.append(f'Struct: {loss_struct_record.show():.4f}')
            if opt.use_ppc_loss:       log_parts.append(f'PPC: {loss_ppc_record.show():.4f}')
            if opt.use_ppd_loss:       log_parts.append(f'PPD: {loss_ppd_record.show():.4f}')
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
    total_dice = 0
    total_images = 0
    datasets = ['CVC-ClinicDB', 'Kvasir', 'CVC-300', 'CVC-ColonDB', 'ETIS-LaribPolypDB']

    for dataset in datasets:
        dataset_dice, n_images = test(model, opt.test_path, dataset, opt.img_size)
        total_dataset_dice += dataset_dice
        total_dice += n_images * dataset_dice
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
        print('##################### Dice score improved from {} to {}'.format(best, total_dataset_dice))
        logging.info('##################### Dice score improved from {} to {}'.format(best, total_dataset_dice))
        best = total_dataset_dice
        torch.save(model.state_dict(), os.path.join(save_path, 'best.pth'))


if __name__ == '__main__':
    dict_plot = {
        'CVC-300': [], 'CVC-ClinicDB': [], 'Kvasir': [],
        'CVC-ColonDB': [], 'ETIS-LaribPolypDB': [], 'test': []
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',        type=int,   default=100)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--batchsize',    type=int,   default=16)
    parser.add_argument('--img_size',     type=int,   default=352)
    parser.add_argument('--clip',         type=float, default=0.5)
    parser.add_argument('--decay_rate',   type=float, default=0.1)
    parser.add_argument('--decay_epoch',  type=int,   default=100)
    parser.add_argument('--train_path',   type=str,   default='/home/U116med/data/polyp/TrainDataset/')
    parser.add_argument('--test_path',    type=str,   default='/home/U116med/data/polyp/TestDataset/')
    parser.add_argument('--train_save',   type=str,   default='models/prototype_cascade/')
    parser.add_argument('--augmentation', default='False')
    parser.add_argument('--device',       type=int,   default=0)
    parser.add_argument('--name',         type=str,   default='prototype_v0_fb33')
    parser.add_argument('--num_prototype',type=int,   default=3,
                        help='每個 class 的 prototype 數量（FG 和 BG 相同）')

    parser.add_argument('--save_epochs', type=int, nargs='+',
                        default=[1, 2, 3, 4, 5, 10, 20])

    parser.add_argument('--use_ce_loss',        type=str2bool, default=False)
    parser.add_argument('--use_ppc_loss',       type=str2bool, default=True)
    parser.add_argument('--use_ppd_loss',       type=str2bool, default=True)
    parser.add_argument('--use_structure_loss', type=str2bool, default=True)

    parser.add_argument('--weight_ppc', type=float, default=1)
    parser.add_argument('--weight_ppd', type=float, default=1)

    opt = parser.parse_args()

    logging.basicConfig(
        filename=f'train_log_{opt.name}.log',
        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p'
    )

    opt.train_save = os.path.join(opt.train_save, opt.name)
    os.makedirs(opt.train_save, exist_ok=True)
    torch.cuda.set_device(opt.device)

    model = Prototype_CASCADE_v0(
        num_classes=2,
        num_prototype=opt.num_prototype
    ).cuda()

    best = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    print(optimizer)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root    = '{}/masks/'.format(opt.train_path)
    train_loader = get_loader(
        image_root, gt_root,
        batchsize=opt.batchsize, trainsize=opt.img_size,
        augmentation=opt.augmentation
    )

    # ── Trackers ─────────────────────────────────────────────────────────
    E = range(1, opt.epoch + 1)
    M = opt.num_prototype
    trackers = {
        # Step-level losses
        'step_ce':     [],
        'step_struct': [],
        'step_ppc':    [],
        'step_ppd':    [],
        'step_total':  [],

        # Step-level prototype competition（similarity_map argmax）
        'step_fg_argmax': {k: [] for k in range(M)},
        'step_bg_argmax': {k: [] for k in range(M)},

        # Epoch-level prototype pixel ratio
        'proto_counts_fg': {e: np.zeros(M) for e in E},
        'proto_counts_bg': {e: np.zeros(M) for e in E},

        # Epoch-level image-level collapse
        'fg_collapse_90':      {e: 0 for e in E},
        'fg_collapse_70':      {e: 0 for e in E},
        'fg_collapse_50':      {e: 0 for e in E},
        'bg_collapse_90':      {e: 0 for e in E},
        'bg_collapse_70':      {e: 0 for e in E},
        'bg_collapse_50':      {e: 0 for e in E},
        'fg_valid_img_count':  {e: 0 for e in E},
        'bg_valid_img_count':  {e: 0 for e in E},
    }

    print("######## Start Training Prototype CASCADE v0 ########")
    for epoch in range(1, opt.epoch + 1):
        train(train_loader, model, optimizer, epoch, opt, trackers)

    # ── Plots ─────────────────────────────────────────────────────────────
    print("######## Generating Plots... ########")
    epochs_list = list(range(1, opt.epoch + 1))
    save_path   = opt.train_save

    # 1. Step-wise Loss Curve
    plt.figure(figsize=(12, 6))
    if opt.use_structure_loss and trackers['step_struct']:
        plt.plot(trackers['step_struct'], label='Structure Loss', alpha=0.8)
    if opt.use_ppc_loss and trackers['step_ppc']:
        plt.plot(trackers['step_ppc'], label='PPC Loss', alpha=0.8)
    if opt.use_ppd_loss and trackers['step_ppd']:
        plt.plot(trackers['step_ppd'], label='PPD Loss', alpha=0.8)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss Value')
    plt.title('Step-wise Loss Curve (v0 · d1 layer)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'step_loss_curve.png'))
    plt.close()

    # 2. Global Prototype Pixel Ratio（d1 layer, similarity_map argmax）
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for k in range(M):
        fg_ratios, bg_ratios = [], []
        for e in epochs_list:
            total_fg = np.sum(trackers['proto_counts_fg'][e])
            total_bg = np.sum(trackers['proto_counts_bg'][e])
            fg_ratios.append(trackers['proto_counts_fg'][e][k] / total_fg if total_fg > 0 else 0)
            bg_ratios.append(trackers['proto_counts_bg'][e][k] / total_bg if total_bg > 0 else 0)
        axes[0].plot(epochs_list, fg_ratios, label=f'FG_{k}', marker='.')
        axes[1].plot(epochs_list, bg_ratios, label=f'BG_{k}', marker='.')

    for ax, title in zip(axes, ['Foreground Prototype Pixel Ratio', 'Background Prototype Pixel Ratio']):
        ax.set_title(f'{title} (d1 layer)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Ratio')
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'prototype_ratio_curve.png'))
    plt.close()

    # 3. Image-Level Collapse（d1 layer）
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    def safe_ratio(num_dict, den_dict, e):
        return num_dict[e] / den_dict[e] if den_dict[e] > 0 else 0

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
        ax.set_title(f'{class_name} Image-Level Collapse (d1 layer)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Ratio of Images')
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'image_level_collapse_curve.png'))
    plt.close()

    # 4. Step-wise Prototype Competition（d1 layer）
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
            ax.plot(range(len(smoothed)), smoothed, label=f'{"FG" if "fg" in argmax_key else "BG"}_{k}', alpha=0.8)
        ax.set_title(f'Step-wise {class_name} Prototype Competition (d1 layer)')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Pixel Selection Ratio')
        ax.set_ylim([-0.05, 1.05])
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'step_competition_curve.png'))
    plt.close()

    print(f"Plots saved to {save_path}")
