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

from lib.networks_dinov3_concat import DINOv3_concat
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter

        
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
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

        results = model(image) # forward
        
        
        res = F.interpolate(results, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze() # apply sigmoid aggregation for binary segmentation
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
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            
            # ---- forward ----
            results= model(images)
            
            # ---- loss function ----
            #loss_P1 = structure_loss(P1, gts)
            #loss_P2 = structure_loss(P2, gts)
            #loss_P3 = structure_loss(P3, gts)
            loss_P4 = structure_loss(results, gts)
            
            #alpha, beta, gamma, zeta = 1., 1., 1., 1. 
            #loss = alpha * loss_P1 + beta * loss_P2 + gamma * loss_P3 + zeta * loss_P4 # current setting is for additive aggregation.
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
                  ' loss: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record.show()))
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
                        default='models/polyp/')

    parser.add_argument('--device', type=int,
                        default=1)
    
    parser.add_argument('--name', type=str,
                        default='baseline_dinov3')

    parser.add_argument('--percentage', type=float,
                        default=1., help='percentage of training')

    parser.add_argument('--backbone', type=str,
                        default='vits16plus')

    parser.add_argument('--freeze', type=bool,
                        default=False)
    
    opt = parser.parse_args()
    logging.basicConfig(filename='train_log_'+opt.name+'_'+opt.backbone+'_fullfinetune.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    opt.train_save = os.path.join(opt.train_save, 'vits16plus/', opt.name)
    # ---- build models ----
    torch.cuda.set_device(opt.device)  # set your gpu device
    
    model = DINOv3_concat(backbone_type=opt.backbone, freeze=opt.freeze)
    model.cuda()
	
    best = 0

    params = filter(lambda p: p.requires_grad, model.parameters())

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
    
