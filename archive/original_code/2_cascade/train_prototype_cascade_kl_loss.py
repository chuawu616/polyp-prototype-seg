import os
import numpy as np
import argparse
from datetime import datetime
import logging

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import CrossEntropyLoss

import matplotlib.pyplot as plt

from lib.networks_prototype import Prototype_CASCADE
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter

def KL_loss(logits_k_class, gt, fg_num, bg_num):
        probs = F.softmax(logits_k_class * 10, dim=1) # (B, K, H, W)        
        mask_fg = gt              # (B, 1, H, W)
        mask_bg = 1.0 - gt        # (B, 1, H, W)
        sum_weight_fg = mask_fg.sum()
        sum_weight_bg = mask_bg.sum()
        loss_kl_total = 0.0       
    
        if sum_weight_fg > 1.0: 
            probs_fg_part = probs[:, 0:fg_num, :, :] 
            p_bar_fg = (probs_fg_part * mask_fg).sum(dim=(0, 2, 3)) / sum_weight_fg
            p_bar_fg = p_bar_fg / (p_bar_fg.sum() + 1e-8)
            t_fg = torch.full_like(p_bar_fg, 1.0 / fg_num)
            kl_fg = torch.sum(p_bar_fg * (torch.log(p_bar_fg + 1e-8) - torch.log(t_fg + 1e-8)))
            loss_kl_total += kl_fg

        if sum_weight_bg > 1.0:
            probs_bg_part = probs[:, fg_num:fg_num+bg_num, :, :]
            p_bar_bg = (probs_bg_part * mask_bg).sum(dim=(0, 2, 3)) / sum_weight_bg
            p_bar_bg = p_bar_bg / (p_bar_bg.sum() + 1e-8)
            t_bg = torch.full_like(p_bar_bg, 1.0 / bg_num)            
            kl_bg = torch.sum(p_bar_bg * (torch.log(p_bar_bg + 1e-8) - torch.log(t_bg + 1e-8)))
            loss_kl_total += kl_bg
            
        return loss_kl_total
    
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

def test(model, path, dataset):
    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, opt.img_size)
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res, _ = model(image)
        res = (res[:, 1:2, :, :] - res[:, 0:1, :, :]) * 10.0
        
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
        # eval Dice
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

    return DSC / num1, num1

def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    global best
    size_rates = [0.75, 1, 1.25] 
    loss_record = AvgMeter()
    
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            
            trainsize = int(round(opt.img_size * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            
            logits, sim_map = model(images)
            
            # 確保 logits 與 gts 尺寸一致
            if logits.shape[-2:] != gts.shape[-2:]:
                logits = F.interpolate(logits, size=gts.shape[-2:], mode='bilinear', align_corners=False)
            
            final_pred = (logits[:, 1:2, :, :] - logits[:, 0:1, :, :]) * 10.0
            
            loss = structure_loss(final_pred, gts)
            
            if epoch <= 5:
                ramp_up_weit = 0
            elif epoch > 5 and epoch < 10:
                ramp_up_weit = opt.kl_weit * ((epoch - 5) / 5)
            else:
                ramp_up_weit = opt.kl_weit
                
            kl_loss = ramp_up_weit * KL_loss(sim_map, gts, opt.fg_num, opt.bg_num)
            
            loss += kl_loss
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)
                
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:0.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))

    # save model 
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path , 'check.pth'))
    
    if (epoch + 1) % 1 == 0:
    	total_dataset_dice = 0
    	total_dice = 0
    	total_images = 0
    	datasets = ['CVC-ClinicDB', 'Kvasir','CVC-300',  'CVC-ColonDB', 'ETIS-LaribPolypDB']
    	for dataset in datasets:
    	    dataset_dice, n_images = test(model, test_path, dataset)
    	    total_dataset_dice += dataset_dice
    	    total_dice += (n_images*dataset_dice)
    	    total_images += n_images
    	    logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
    	    print(dataset, ': ', dataset_dice)
    	    dict_plot[dataset].append(dataset_dice)
    	meandice = total_dice/total_images
    	total_dataset_dice = total_dataset_dice/len(datasets)
    	dict_plot['test'].append(meandice)
    	print('mdice: {:.04f}'.format(total_dataset_dice))
    	print('Validation dice score: {}'.format(meandice))
    	logging.info('Validation dice score: {}'.format(meandice))
    	if total_dataset_dice > best:
            print('##################### Dice score improved from {} to {}'.format(best, total_dataset_dice))
            logging.info('##################### Dice score improved from {} to {}'.format(best, total_dataset_dice))
            best = total_dataset_dice
            torch.save(model.state_dict(), os.path.join(save_path , 'best.pth'))
            # torch.save(model.state_dict(), os.path.join(save_path , 'best_{:03d}.pth'.format(epoch)))
    
if __name__ == '__main__':
    dict_plot = {'CVC-300':[], 'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-ColonDB':[], 'ETIS-LaribPolypDB':[], 'test':[]}
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    ##################model_name#############################
    ###############################################
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
    
    parser.add_argument('--augmentation', default='False')
    
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--name', type=str, default='prototype_kl_loss_0.1_fb_88_v3')

    parser.add_argument('--fg_num', type=int, default=8)

    parser.add_argument('--bg_num', type=int, default=8)

    parser.add_argument('--kl_weit', type=float, default=0.1)
    
    opt = parser.parse_args()
    logging.basicConfig(filename='train_log_'+opt.name+'.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    opt.train_save = os.path.join(opt.train_save, opt.name)
    
    torch.cuda.set_device(opt.device)
    
    model = Prototype_CASCADE(num_fg=opt.fg_num, num_bg=opt.bg_num, encoder_path=None).cuda()
    
    best = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    print(optimizer)
    
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)
    
    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.img_size, augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("########Start Training Prototype CASCADE########")
    for epoch in range(1, opt.epoch + 1):
        #adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, opt.test_path)