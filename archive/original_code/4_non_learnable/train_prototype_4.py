import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from lib.networks_prototype_v3 import Prototype_CASCADE_v3
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
    def __init__(self, ignore_label=-1):
        super(PPC, self).__init__()
        self.ignore_label = ignore_label

    def forward(self, contrast_logits, contrast_target):
        loss_ppc = F.cross_entropy(contrast_logits, contrast_target.long(), ignore_index=self.ignore_label)
        return loss_ppc


class PPD(nn.Module):
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
        loss_ppd = (1 - logits).pow(2).mean()
        return loss_ppd


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
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
    
    criterion_ce = nn.CrossEntropyLoss(ignore_index=-1).cuda()
    criterion_ppc = PPC().cuda()
    criterion_ppd = PPD().cuda()

    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts = pack
        images = images.cuda()
        gts = gts.cuda()
        gts_idx = gts.squeeze(1).long() 
        
        out_dict = model(images, gt_semantic_seg=gts_idx)
        seg_logits = out_dict['seg']           
        
        loss = torch.tensor(0.0).cuda()
        
        if opt.use_ce_loss:
            loss_ce = criterion_ce(seg_logits, gts_idx)
            loss = loss + loss_ce
            loss_ce_record.update(loss_ce.data, opt.batchsize)
            trackers['step_ce'].append(loss_ce.item())
            
        if opt.use_structure_loss:
            fg_logits = ((seg_logits[:, 1, :, :] - seg_logits[:, 0, :, :])*10.0).unsqueeze(1)
            loss_struct = structure_loss(fg_logits, gts)
            loss = loss + loss_struct
            loss_struct_record.update(loss_struct.data, opt.batchsize)
            trackers['step_struct'].append(loss_struct.item())

        logits_node = out_dict.get('logits_node', None)
        labels_node = out_dict.get('labels_node', None)
        logits_d1 = out_dict.get('logits_d1', None)
        labels_d1 = out_dict.get('labels_d1', None)
        similarity_map = out_dict.get('similarity_map', None)
            
        if opt.use_ppc_loss and logits_node is not None and labels_node is not None:
            loss_ppc = criterion_ppc(logits_node, labels_node)
            loss = loss + (opt.weight_ppc * loss_ppc)
            loss_ppc_record.update(loss_ppc.data, opt.batchsize)
            trackers['step_ppc'].append(loss_ppc.item())
                
        if opt.use_ppd_loss and logits_d1 is not None and labels_d1 is not None:
            loss_ppd = criterion_ppd(logits_d1, labels_d1)
            loss = loss + (opt.weight_ppd * loss_ppd)
            loss_ppd_record.update(loss_ppd.data, opt.batchsize)
            trackers['step_ppd'].append(loss_ppd.item())

        if loss.requires_grad:
            loss.backward()
            clip_gradient(optimizer, opt.clip) 
            optimizer.step()
            loss_record.update(loss.data, opt.batchsize)
            trackers['step_total'].append(loss.item())

        # ---------------- Node Level (Target Node) Statistics ----------------
        if labels_node is not None:
            labels_np = labels_node.detach().cpu().numpy()
            B = images.shape[0]
            labels_batch = labels_np.reshape(B, -1)
            
            for b_idx in range(B):
                img_labels = labels_batch[b_idx]
                
                # Foreground Node Collapse
                fg_mask = (img_labels >= opt.bg_num) & (img_labels < opt.bg_num + opt.fg_num)
                fg_pixels = img_labels[fg_mask]
                if len(fg_pixels) > 0:
                    counts = np.bincount(fg_pixels.astype(int) - opt.bg_num, minlength=opt.fg_num)
                    max_ratio = counts.max() / len(fg_pixels)
                    if max_ratio > 0.50: trackers['fg_collapse_50'][epoch] += 1
                    if max_ratio > 0.70: trackers['fg_collapse_70'][epoch] += 1
                    if max_ratio > 0.90: trackers['fg_collapse_90'][epoch] += 1
                    trackers['fg_valid_img_count'][epoch] += 1
                
                # Background Node Collapse
                bg_mask = (img_labels >= 0) & (img_labels < opt.bg_num)
                bg_pixels = img_labels[bg_mask]
                if len(bg_pixels) > 0:
                    counts = np.bincount(bg_pixels.astype(int), minlength=opt.bg_num)
                    max_ratio = counts.max() / len(bg_pixels)
                    if max_ratio > 0.50: trackers['bg_collapse_50'][epoch] += 1
                    if max_ratio > 0.70: trackers['bg_collapse_70'][epoch] += 1
                    if max_ratio > 0.90: trackers['bg_collapse_90'][epoch] += 1
                    trackers['bg_valid_img_count'][epoch] += 1

            for k in range(opt.bg_num):
                trackers['proto_counts_bg'][epoch][k] += np.sum(labels_np == k)
            for k in range(opt.fg_num):
                trackers['proto_counts_fg'][epoch][k] += np.sum(labels_np == (opt.bg_num + k))

        # ---------------- D1 Level Statistics (基於 Similarity Map) ----------------
        if similarity_map is not None:
            pred_k_mask_d1 = similarity_map.argmax(dim=1).detach()
            B = images.shape[0]
            labels_d1_batch = pred_k_mask_d1.cpu().numpy().reshape(B, -1)
            
            # --- 1. Step-level 競爭追蹤 ---
            # Foreground Step Tracking
            fg_mask_step = (pred_k_mask_d1 >= opt.bg_num) & (pred_k_mask_d1 < opt.bg_num + opt.fg_num)
            fg_pixels_step = pred_k_mask_d1[fg_mask_step]
            if len(fg_pixels_step) > 0:
                fg_counts = torch.bincount(fg_pixels_step - opt.bg_num, minlength=opt.fg_num).cpu().numpy()
            else:
                fg_counts = np.zeros(opt.fg_num)
            for k in range(opt.fg_num):
                trackers['step_d1_fg_argmax'][k].append(fg_counts[k])
                
            # Background Step Tracking
            bg_mask_step = (pred_k_mask_d1 >= 0) & (pred_k_mask_d1 < opt.bg_num)
            bg_pixels_step = pred_k_mask_d1[bg_mask_step]
            if len(bg_pixels_step) > 0:
                bg_counts = torch.bincount(bg_pixels_step, minlength=opt.bg_num).cpu().numpy()
            else:
                bg_counts = np.zeros(opt.bg_num)
            for k in range(opt.bg_num):
                trackers['step_d1_bg_argmax'][k].append(bg_counts[k])
            
            # --- 2. Epoch-level Image Collapse & Global Ratio ---
            for b_idx in range(B):
                img_labels_d1 = labels_d1_batch[b_idx]
                
                # Foreground D1 Collapse
                fg_mask_d1 = (img_labels_d1 >= opt.bg_num) & (img_labels_d1 < opt.bg_num + opt.fg_num)
                fg_pixels_d1 = img_labels_d1[fg_mask_d1]
                if len(fg_pixels_d1) > 0:
                    counts = np.bincount(fg_pixels_d1.astype(int) - opt.bg_num, minlength=opt.fg_num)
                    max_ratio = counts.max() / len(fg_pixels_d1)
                    if max_ratio > 0.50: trackers['d1_fg_collapse_50'][epoch] += 1
                    if max_ratio > 0.70: trackers['d1_fg_collapse_70'][epoch] += 1
                    if max_ratio > 0.90: trackers['d1_fg_collapse_90'][epoch] += 1
                    trackers['d1_fg_valid_img_count'][epoch] += 1
                
                # Background D1 Collapse
                bg_mask_d1 = (img_labels_d1 >= 0) & (img_labels_d1 < opt.bg_num)
                bg_pixels_d1 = img_labels_d1[bg_mask_d1]
                if len(bg_pixels_d1) > 0:
                    counts = np.bincount(bg_pixels_d1.astype(int), minlength=opt.bg_num)
                    max_ratio = counts.max() / len(bg_pixels_d1)
                    if max_ratio > 0.50: trackers['d1_bg_collapse_50'][epoch] += 1
                    if max_ratio > 0.70: trackers['d1_bg_collapse_70'][epoch] += 1
                    if max_ratio > 0.90: trackers['d1_bg_collapse_90'][epoch] += 1
                    trackers['d1_bg_valid_img_count'][epoch] += 1

            for k in range(opt.bg_num):
                trackers['d1_proto_counts_bg'][epoch][k] += np.sum(labels_d1_batch == k)
            for k in range(opt.fg_num):
                trackers['d1_proto_counts_fg'][epoch][k] += np.sum(labels_d1_batch == (opt.bg_num + k))
        
        if i % 20 == 0 or i == total_step:
            log_parts = [f'Epoch [{epoch:03d}/{opt.epoch:03d}]', 
                         f'Step [{i:04d}/{total_step:04d}]', 
                         f'Loss: {loss_record.show():.4f}']
            
            if opt.use_ce_loss:
                log_parts.append(f'CE: {loss_ce_record.show():.4f}')
            if opt.use_structure_loss:
                log_parts.append(f'Struct: {loss_struct_record.show():.4f}')
            if opt.use_ppc_loss:
                log_parts.append(f'PPC: {loss_ppc_record.show():.4f}')
            if opt.use_ppd_loss:
                log_parts.append(f'PPD: {loss_ppd_record.show():.4f}')
                    
            log_info = ', '.join(log_parts)
                
            print(log_info)
            logging.info(log_info)
            
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    torch.save(model.state_dict(), os.path.join(save_path , 'check.pth'))
    
    if epoch in opt.save_epochs:
        torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch}.pth'))
        logging.info(f'Saved specified epoch weight: epoch_{epoch}.pth')
    
    if (epoch + 1) % 1 == 0:
        total_dataset_dice = 0
        total_dice = 0
        total_images = 0
        datasets = ['CVC-ClinicDB', 'Kvasir', 'CVC-300', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
            
        for dataset in datasets:
            dataset_dice, n_images = test(model, opt.test_path, dataset, opt.img_size)
            total_dataset_dice += dataset_dice
            total_dice += (n_images * dataset_dice)
            total_images += n_images
            logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
            print(dataset, ': ', dataset_dice)
            dict_plot[dataset].append(dataset_dice)
                
        meandice = total_dice / total_images
        total_dataset_dice = total_dataset_dice / len(datasets)
        dict_plot['test'].append(meandice)
            
        print('mdice: {:.04f}'.format(total_dataset_dice))
        print('Validation dice score: {}'.format(meandice))
        logging.info('Validation dice score: {}'.format(meandice))
            
        if total_dataset_dice > best:
            print('##################### Dice score improved from {} to {}'.format(best, total_dataset_dice))
            logging.info('##################### Dice score improved from {} to {}'.format(best, total_dataset_dice))
            best = total_dataset_dice
            torch.save(model.state_dict(), os.path.join(save_path, 'best.pth'))


if __name__ == '__main__':
    dict_plot = {'CVC-300':[], 'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-ColonDB':[], 'ETIS-LaribPolypDB':[], 'test':[]}
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=352)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--decay_epoch', type=int, default=100)
    parser.add_argument('--train_path', type=str, default='/home/U116med/data/polyp/TrainDataset/')
    parser.add_argument('--test_path', type=str, default='/home/U116med/data/polyp/TestDataset/')
    parser.add_argument('--train_save', type=str, default='models/prototype_cascade/')
    parser.add_argument('--augmentation', default='False')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--name', type=str, default='prototypev3_fb55_x4')
    parser.add_argument('--target_node', type=str, default='x4')
    parser.add_argument('--fg_num', type=int, default=5)
    parser.add_argument('--bg_num', type=int, default=5)
    
    parser.add_argument('--save_epochs', type=int, nargs='+', default=[1, 2, 3, 4, 5, 10, 20])
    
    parser.add_argument('--use_ce_loss', type=str2bool, default=False)
    parser.add_argument('--use_ppc_loss', type=str2bool, default=True)
    parser.add_argument('--use_ppd_loss', type=str2bool, default=True)
    parser.add_argument('--use_structure_loss', type=str2bool, default=True)
    
    parser.add_argument('--weight_ppc', type=float, default=0.5)
    parser.add_argument('--weight_ppd', type=float, default=0.5)

    parser.add_argument('--pretrained_model_path', type=str, default='/home/U116med/wch_code/non_learnable/models/polyp/baseline_100epoch/best.pth')
    parser.add_argument('--kmeans_center_path', type=str, default='/home/U116med/wch_code/non_learnable/kmeans_center/kmeans_prototypes_dd4_ch512.pth')

    opt = parser.parse_args()
    
    logging.basicConfig(filename='train_log_'+opt.name+'.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    
    opt.train_save = os.path.join(opt.train_save, opt.name)
    if not os.path.exists(opt.train_save):
        os.makedirs(opt.train_save)
    
    torch.cuda.set_device(opt.device)
    
    model = Prototype_CASCADE_v3(num_prototype=opt.fg_num,
                                 pretrained_model_path=opt.pretrained_model_path,
                                 kmeans_center_path=opt.kmeans_center_path,
                                 target_node=opt.target_node).cuda()
    
    best = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    print(optimizer)
    
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)
    
    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.img_size, augmentation=opt.augmentation)
    total_step = len(train_loader)

    trackers = {
        'step_ce': [], 'step_struct': [], 'step_ppc': [], 'step_ppd': [], 'step_total': [],
        'step_d1_fg_argmax': {k: [] for k in range(opt.fg_num)},
        'step_d1_bg_argmax': {k: [] for k in range(opt.bg_num)},
        # Node trackers
        'proto_counts_bg': {e: np.zeros(opt.bg_num) for e in range(1, opt.epoch + 1)},
        'proto_counts_fg': {e: np.zeros(opt.fg_num) for e in range(1, opt.epoch + 1)},
        'fg_collapse_90': {e: 0 for e in range(1, opt.epoch + 1)},
        'fg_collapse_70': {e: 0 for e in range(1, opt.epoch + 1)},
        'fg_collapse_50': {e: 0 for e in range(1, opt.epoch + 1)},
        'bg_collapse_90': {e: 0 for e in range(1, opt.epoch + 1)},
        'bg_collapse_70': {e: 0 for e in range(1, opt.epoch + 1)},
        'bg_collapse_50': {e: 0 for e in range(1, opt.epoch + 1)},
        'fg_valid_img_count': {e: 0 for e in range(1, opt.epoch + 1)},
        'bg_valid_img_count': {e: 0 for e in range(1, opt.epoch + 1)},
        
        # D1 trackers
        'd1_proto_counts_bg': {e: np.zeros(opt.bg_num) for e in range(1, opt.epoch + 1)},
        'd1_proto_counts_fg': {e: np.zeros(opt.fg_num) for e in range(1, opt.epoch + 1)},
        'd1_fg_collapse_90': {e: 0 for e in range(1, opt.epoch + 1)},
        'd1_fg_collapse_70': {e: 0 for e in range(1, opt.epoch + 1)},
        'd1_fg_collapse_50': {e: 0 for e in range(1, opt.epoch + 1)},
        'd1_bg_collapse_90': {e: 0 for e in range(1, opt.epoch + 1)},
        'd1_bg_collapse_70': {e: 0 for e in range(1, opt.epoch + 1)},
        'd1_bg_collapse_50': {e: 0 for e in range(1, opt.epoch + 1)},
        'd1_fg_valid_img_count': {e: 0 for e in range(1, opt.epoch + 1)},
        'd1_bg_valid_img_count': {e: 0 for e in range(1, opt.epoch + 1)}
    }

    print("######## Start Training Prototype CASCADE v3 ########")
    for epoch in range(1, opt.epoch + 1):
        train(train_loader, model, optimizer, epoch, opt, trackers)
        
    print("######## Training Completed. Generating Plots... ########")
    epochs_list = list(range(1, opt.epoch + 1))
    
    # 1. Step-wise Loss Curve
    plt.figure(figsize=(12, 6))
    if opt.use_structure_loss:
        plt.plot(trackers['step_struct'], label='Structure Loss', alpha=0.8)
    if opt.use_ppc_loss:
        plt.plot(trackers['step_ppc'], label='PPC Loss', alpha=0.8)
    if opt.use_ppd_loss:
        plt.plot(trackers['step_ppd'], label='PPD Loss', alpha=0.8)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss Value')
    plt.title('Step-wise Loss Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(opt.train_save, 'step_loss_curve.png'))
    plt.close()

    # 2. Global Prototype Pixel Ratio (Node Level)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for k in range(opt.fg_num):
        ratios = []
        for e in epochs_list:
            total_fg = np.sum(trackers['proto_counts_fg'][e])
            ratio = trackers['proto_counts_fg'][e][k] / total_fg if total_fg > 0 else 0
            ratios.append(ratio)
        axes[0].plot(epochs_list, ratios, label=f'FG_{k}', marker='.')
    axes[0].set_title('Global Foreground Prototype Pixel Ratio (Target Node)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Ratio')
    axes[0].set_ylim([-0.05, 1.05])
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()
    
    for k in range(opt.bg_num):
        ratios = []
        for e in epochs_list:
            total_bg = np.sum(trackers['proto_counts_bg'][e])
            ratio = trackers['proto_counts_bg'][e][k] / total_bg if total_bg > 0 else 0
            ratios.append(ratio)
        axes[1].plot(epochs_list, ratios, label=f'BG_{k}', marker='.')
    axes[1].set_title('Global Background Prototype Pixel Ratio (Target Node)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Ratio')
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(opt.train_save, 'prototype_ratio_curve.png'))
    plt.close()

    # 3. Global Prototype Pixel Ratio (D1 Level)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for k in range(opt.fg_num):
        ratios = []
        for e in epochs_list:
            total_fg = np.sum(trackers['d1_proto_counts_fg'][e])
            ratio = trackers['d1_proto_counts_fg'][e][k] / total_fg if total_fg > 0 else 0
            ratios.append(ratio)
        axes[0].plot(epochs_list, ratios, label=f'FG_{k}', marker='.')
    axes[0].set_title('Global Foreground Prototype Pixel Ratio (D1 Node)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Ratio')
    axes[0].set_ylim([-0.05, 1.05])
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()
    
    for k in range(opt.bg_num):
        ratios = []
        for e in epochs_list:
            total_bg = np.sum(trackers['d1_proto_counts_bg'][e])
            ratio = trackers['d1_proto_counts_bg'][e][k] / total_bg if total_bg > 0 else 0
            ratios.append(ratio)
        axes[1].plot(epochs_list, ratios, label=f'BG_{k}', marker='.')
    axes[1].set_title('Global Background Prototype Pixel Ratio (D1 Node)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Ratio')
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(opt.train_save, 'd1_prototype_ratio_curve.png'))
    plt.close()
    
    # 4. Multi-level Image Collapse (Target Node)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fg_collapse_90_ratios = [trackers['fg_collapse_90'][e] / trackers['fg_valid_img_count'][e] if trackers['fg_valid_img_count'][e] > 0 else 0 for e in epochs_list]
    fg_collapse_70_ratios = [trackers['fg_collapse_70'][e] / trackers['fg_valid_img_count'][e] if trackers['fg_valid_img_count'][e] > 0 else 0 for e in epochs_list]
    fg_collapse_50_ratios = [trackers['fg_collapse_50'][e] / trackers['fg_valid_img_count'][e] if trackers['fg_valid_img_count'][e] > 0 else 0 for e in epochs_list]
    
    axes[0].plot(epochs_list, fg_collapse_90_ratios, label='>90% Dominance', marker='o', color='darkred')
    axes[0].plot(epochs_list, fg_collapse_70_ratios, label='>70% Dominance', marker='s', color='red', alpha=0.7)
    axes[0].plot(epochs_list, fg_collapse_50_ratios, label='>50% Dominance', marker='^', color='salmon', alpha=0.7)
    axes[0].set_title('Foreground Image-Level Collapse (Target Node)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Ratio of Images')
    axes[0].set_ylim([-0.05, 1.05])
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()
    
    bg_collapse_90_ratios = [trackers['bg_collapse_90'][e] / trackers['bg_valid_img_count'][e] if trackers['bg_valid_img_count'][e] > 0 else 0 for e in epochs_list]
    bg_collapse_70_ratios = [trackers['bg_collapse_70'][e] / trackers['bg_valid_img_count'][e] if trackers['bg_valid_img_count'][e] > 0 else 0 for e in epochs_list]
    bg_collapse_50_ratios = [trackers['bg_collapse_50'][e] / trackers['bg_valid_img_count'][e] if trackers['bg_valid_img_count'][e] > 0 else 0 for e in epochs_list]
    
    axes[1].plot(epochs_list, bg_collapse_90_ratios, label='>90% Dominance', marker='o', color='darkblue')
    axes[1].plot(epochs_list, bg_collapse_70_ratios, label='>70% Dominance', marker='s', color='blue', alpha=0.7)
    axes[1].plot(epochs_list, bg_collapse_50_ratios, label='>50% Dominance', marker='^', color='cornflowerblue', alpha=0.7)
    axes[1].set_title('Background Image-Level Collapse (Target Node)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Ratio of Images')
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(opt.train_save, 'image_level_collapse_curve.png'))
    plt.close()

    # 5. Multi-level Image Collapse (D1 Level)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    d1_fg_collapse_90_ratios = [trackers['d1_fg_collapse_90'][e] / trackers['d1_fg_valid_img_count'][e] if trackers['d1_fg_valid_img_count'][e] > 0 else 0 for e in epochs_list]
    d1_fg_collapse_70_ratios = [trackers['d1_fg_collapse_70'][e] / trackers['d1_fg_valid_img_count'][e] if trackers['d1_fg_valid_img_count'][e] > 0 else 0 for e in epochs_list]
    d1_fg_collapse_50_ratios = [trackers['d1_fg_collapse_50'][e] / trackers['d1_fg_valid_img_count'][e] if trackers['d1_fg_valid_img_count'][e] > 0 else 0 for e in epochs_list]
    
    axes[0].plot(epochs_list, d1_fg_collapse_90_ratios, label='>90% Dominance', marker='o', color='darkred')
    axes[0].plot(epochs_list, d1_fg_collapse_70_ratios, label='>70% Dominance', marker='s', color='red', alpha=0.7)
    axes[0].plot(epochs_list, d1_fg_collapse_50_ratios, label='>50% Dominance', marker='^', color='salmon', alpha=0.7)
    axes[0].set_title('Foreground Image-Level Collapse (D1 Node)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Ratio of Images')
    axes[0].set_ylim([-0.05, 1.05])
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()
    
    d1_bg_collapse_90_ratios = [trackers['d1_bg_collapse_90'][e] / trackers['d1_bg_valid_img_count'][e] if trackers['d1_bg_valid_img_count'][e] > 0 else 0 for e in epochs_list]
    d1_bg_collapse_70_ratios = [trackers['d1_bg_collapse_70'][e] / trackers['d1_bg_valid_img_count'][e] if trackers['d1_bg_valid_img_count'][e] > 0 else 0 for e in epochs_list]
    d1_bg_collapse_50_ratios = [trackers['d1_bg_collapse_50'][e] / trackers['d1_bg_valid_img_count'][e] if trackers['d1_bg_valid_img_count'][e] > 0 else 0 for e in epochs_list]
    
    axes[1].plot(epochs_list, d1_bg_collapse_90_ratios, label='>90% Dominance', marker='o', color='darkblue')
    axes[1].plot(epochs_list, d1_bg_collapse_70_ratios, label='>70% Dominance', marker='s', color='blue', alpha=0.7)
    axes[1].plot(epochs_list, d1_bg_collapse_50_ratios, label='>50% Dominance', marker='^', color='cornflowerblue', alpha=0.7)
    axes[1].set_title('Background Image-Level Collapse (D1 Node)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Ratio of Images')
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(opt.train_save, 'd1_image_level_collapse_curve.png'))
    plt.close()

    # 6. Step-wise Prototype Competition (D1 Level)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    def moving_average(a, n=50):
        if len(a) < n: 
            return a
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    # FG Step Curve
    step_fg_counts = np.array([trackers['step_d1_fg_argmax'][k] for k in range(opt.fg_num)])
    step_fg_totals = step_fg_counts.sum(axis=0) + 1e-8
    step_fg_ratios = step_fg_counts / step_fg_totals

    for k in range(opt.fg_num):
        smoothed_ratio = moving_average(step_fg_ratios[k])
        axes[0].plot(range(len(smoothed_ratio)), smoothed_ratio, label=f'FG_{k}', alpha=0.8)

    axes[0].set_title('Step-wise Foreground Prototype Competition (D1)')
    axes[0].set_xlabel('Training Steps')
    axes[0].set_ylabel('Pixel Selection Ratio')
    axes[0].set_ylim([-0.05, 1.05])
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # BG Step Curve
    step_bg_counts = np.array([trackers['step_d1_bg_argmax'][k] for k in range(opt.bg_num)])
    step_bg_totals = step_bg_counts.sum(axis=0) + 1e-8
    step_bg_ratios = step_bg_counts / step_bg_totals

    for k in range(opt.bg_num):
        smoothed_ratio = moving_average(step_bg_ratios[k])
        axes[1].plot(range(len(smoothed_ratio)), smoothed_ratio, label=f'BG_{k}', alpha=0.8)

    axes[1].set_title('Step-wise Background Prototype Competition (D1)')
    axes[1].set_xlabel('Training Steps')
    axes[1].set_ylabel('Pixel Selection Ratio')
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(opt.train_save, 'step_competition_curve.png'))
    plt.close()
    
    print(f"Plots saved to {opt.train_save}")