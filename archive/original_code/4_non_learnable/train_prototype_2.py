import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.networks_prototype_v2 import Prototype_CASCADE_v2
from lib.networks_prototype_dinov3_v2 import Prototype_DINOv3_v2
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


def train(train_loader, model, optimizer, epoch, opt):
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
            
        if opt.use_structure_loss:
            fg_logits = ((seg_logits[:, 1, :, :] - seg_logits[:, 0, :, :])*10.0).unsqueeze(1)
            loss_struct = structure_loss(fg_logits, gts)
            loss = loss + loss_struct
            loss_struct_record.update(loss_struct.data, opt.batchsize)

        contrast_logits = out_dict['logits']   
        contrast_target = out_dict['target']   
            
        if opt.use_ppc_loss:
            loss_ppc = criterion_ppc(contrast_logits, contrast_target)
            loss = loss + (opt.weight_ppc * loss_ppc)
            loss_ppc_record.update(loss_ppc.data, opt.batchsize)
                
        if opt.use_ppd_loss:
            loss_ppd = criterion_ppd(contrast_logits, contrast_target)
            loss = loss + (opt.weight_ppd * loss_ppd)
            loss_ppd_record.update(loss_ppd.data, opt.batchsize)

        if loss.requires_grad:
            loss.backward()
            clip_gradient(optimizer, opt.clip) 
            optimizer.step()
            loss_record.update(loss.data, opt.batchsize)
        
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
            
    # save model 
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path , 'check.pth'))
    
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
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=352)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--decay_epoch', type=int, default=100)
    parser.add_argument('--train_path', type=str, default='/home/U116med/data/polyp/TrainDataset/')
    parser.add_argument('--test_path', type=str, default='/home/U116med/data/polyp/TestDataset/')
    parser.add_argument('--train_save', type=str, default='models/prototype_cascade/')
    parser.add_argument('--augmentation', default='True')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--name', type=str, default='prototype_fb55_pretrained100_aug_str_replace_ce')
    parser.add_argument('--fg_num', type=int, default=5)
    parser.add_argument('--bg_num', type=int, default=5)
    
    # 損失函數控制選項
    parser.add_argument('--use_ce_loss', type=str2bool, default=False)
    parser.add_argument('--use_ppc_loss', type=str2bool, default=True)
    parser.add_argument('--use_ppd_loss', type=str2bool, default=True)
    parser.add_argument('--use_structure_loss', type=str2bool, default=True)
    
    parser.add_argument('--weight_ppc', type=float, default=0.01)
    parser.add_argument('--weight_ppd', type=float, default=0.01)

    parser.add_argument('--pretrained_model_path', type=str, default='/home/U116med/wch_code/non_learnable/models/polyp/baseline_100epoch/best.pth')
    parser.add_argument('--kmeans_center_path', type=str, default='/home/U116med/wch_code/non_learnable/kmeans_prototypes_100.pth')

    opt = parser.parse_args()
    
    logging.basicConfig(filename='train_log_'+opt.name+'.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    
    opt.train_save = os.path.join(opt.train_save, opt.name)
    
    torch.cuda.set_device(opt.device)
    
    model = Prototype_CASCADE_v2(num_prototype=opt.fg_num, 
                                 pretrained_model_path=opt.pretrained_model_path,
                                 kmeans_center_path=opt.kmeans_center_path).cuda()
    
    best = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    print(optimizer)
    
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)
    
    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.img_size, augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("######## Start Training Prototype CASCADE v2 ########")
    for epoch in range(1, opt.epoch + 1):
        train(train_loader, model, optimizer, epoch, opt)