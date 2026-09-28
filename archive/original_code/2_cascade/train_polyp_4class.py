import os
import numpy as np
import random
import argparse
from datetime import datetime
import logging

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import CrossEntropyLoss

import matplotlib.pyplot as plt

from lib.networks import PVT_CASCADE
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter

        
def structure_loss(pred, mask):
    b, c, h, w = pred.shape
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    
    kernel_size = 2*(h//100)+1
    edge = 2*torch.abs(F.avg_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size//2) - mask)
    pos_edge = (edge>0)*mask
    neg_edge = (edge>0)*(1-mask)
    pos_mask = mask - pos_edge
    neg_mask = (1-mask) - neg_edge
    one_hot_target = torch.cat([pos_mask, pos_edge, neg_mask, neg_edge], dim=1)

    pos_a = torch.clamp(2*F.avg_pool2d(pos_mask, kernel_size=kernel_size, stride=1, padding=kernel_size//2)-1, min=0., max=1.)
    neg_a = torch.clamp(2*F.avg_pool2d(neg_mask, kernel_size=kernel_size, stride=1, padding=kernel_size//2, count_include_pad=False)-1, min=0., max=1.)
    weit1 = torch.clamp(pos_a+neg_a+edge, min=0., max=1.)
    weit1 = ((weit1-weit1.min())/(weit1.max()-weit1.min())).detach()
    #weit1 = torch.ones_like(mask)

    wbce = F.cross_entropy(pred, one_hot_target, reduction='none').unsqueeze(1)
    wbce = (weit1 * wbce).sum(dim=(2, 3)) / weit1.sum(dim=(2, 3))
    pred_softmax = F.softmax(pred, dim=1)
    pred = torch.sum(pred_softmax[:,0:2], dim=1, keepdim=True)

    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean(), wbce.mean(), wiou.mean()

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

        res1, res2, res3, res4 = model(image) # forward
        
        
        res = F.upsample(res4, size=gt.shape, mode='bilinear', align_corners=False) # additive aggregation and upsampling
        res = res.argmax(dim=1, keepdim=True)
        res = (1.*(res==0)+1.*(res==1)).data.cpu().numpy().squeeze() # apply sigmoid aggregation for binary segmentation
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

def train(train_loader, model, optimizer, epoch, test_path, model_name = 'PVT-CASCADE'):
    model.train()
    global best
    size_rates = [0.75, 1, 1.25] 
    loss_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.img_size * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            
            # ---- forward ----
            P1, P2, P3, P4= model(images)
            # ---- loss function ----
            loss_P4, loss_ce, loss_iou = structure_loss(P4, gts)
            loss = loss_P4
            
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)
                
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' loss: {:0.4f}], loss_ce: {:0.4f}, loss_iou: {:0.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record.show(), loss_ce.item(), loss_iou.item()))
    # save model 
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path , 'check.pth'))
    # choose the best model

    global dict_plot
   
    if (epoch + 1) % 1 == 0:
        total_dataset_dice = 0
        total_dice = 0
        total_images = 0
        datasets = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
        for dataset in datasets:
            dataset_dice, n_images = test(model, test_path, dataset)
            total_dataset_dice += dataset_dice
            total_dice += (n_images*dataset_dice)
            total_images += n_images
            logging.info('epoch: {}, dataset: {}, dice: {:.04f}'.format(epoch, dataset, dataset_dice))
            print('dataset: {}, dice: {:.04f}'.format(dataset, dataset_dice))
            dict_plot[dataset].append(dataset_dice)
        meandice = total_dice/total_images
        total_dataset_dice = total_dataset_dice/len(datasets)
        dict_plot['test'].append(meandice)
        print('mdice: {:.04f}'.format(total_dataset_dice))
        print('Validation dice score: {:.04f}'.format(meandice))
        logging.info('Validation dice score: {:.04f}'.format(meandice))
        if total_dataset_dice > best:
            print('##################### Dice score improved from {:.04f} to {:.04f}'.format(best, total_dataset_dice))
            logging.info('##################### Dice score improved from {:.04f} to {:.04f}'.format(best, total_dataset_dice))
            best = total_dataset_dice
            torch.save(model.state_dict(), os.path.join(save_path , 'best.pth'))
            torch.save(model.state_dict(), os.path.join(save_path , 'best_{:03d}.pth'.format(epoch)))
    
if __name__ == '__main__':
    dict_plot = {'CVC-300':[], 'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-ColonDB':[], 'ETIS-LaribPolypDB':[], 'test':[]}
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    ##################model_name#############################
    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')

    parser.add_argument('--img_size', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=100, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='/home/U116med/data/polyp/TrainDataset/',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='/home/U116med/data/polyp/TestDataset/',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='./weights/polyp/')

    parser.add_argument('--device', type=int,
                        default=0)
    
    parser.add_argument('--name', type=str,
                        default='GCSA')
    
    parser.add_argument('--edge', action='store_true')
    parser.add_argument('--percentage', type=float,
                        default=1., help='percentage of training')
    
    opt = parser.parse_args()
    logging.basicConfig(filename='train_log_'+opt.name+'.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    opt.train_save = os.path.join(opt.train_save, opt.name)

    # random seed
    # torch.manual_seed(123)
    # torch.cuda.manual_seed_all(123)
    # np.random.seed(123)
    # random.seed(123)
    # torch.backends.cudnn.enabled=False
    # torch.backends.cudnn.deterministic=True
    
    # ---- build models ----
    torch.cuda.set_device(opt.device)  # set your gpu device
    model = PVT_CASCADE(n_class=4)
    model.cuda()
	
    best = 0

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.img_size,
                              augmentation=opt.augmentation, percentage=opt.percentage)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch):
        #adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, opt.test_path, model_name = opt.name)
    
