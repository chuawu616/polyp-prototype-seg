import os
import numpy as np
import argparse
from datetime import datetime
import logging

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from lib.networks_prototype_adaptive import DINOv3_Prototype_Adaptive
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred_sigmoid = torch.sigmoid(pred)
    inter = ((pred_sigmoid * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sigmoid + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

def test(model, path, dataset, img_size):
    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    
    test_loader = test_dataset(image_root, gt_root, img_size)
    num1 = test_loader.size
    DSC = 0.0
    
    with torch.no_grad():
        for i in range(num1):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            
            res, _, _ = model(image)
            
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            
            pred = np.where(res >= 0.5, 1, 0)
            target = np.where(gt >= 0.5, 1, 0)
            
            smooth = 1
            input_flat = np.reshape(pred, (-1))
            target_flat = np.reshape(target, (-1))
            intersection = (input_flat * target_flat)
            
            dice = (2 * intersection.sum() + smooth) / (pred.sum() + target.sum() + smooth)
            DSC += float(dice)

    return DSC / num1, num1

def train(train_loader, model, optimizer, epoch, test_path, opt):
    model.train()
    global best
    size_rates = [0.75, 1, 1.25] 
    loss_record = AvgMeter()
    total_step = len(train_loader)
    
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            images, gts = pack
            images = images.cuda()
            gts = gts.cuda()
            
            trainsize = int(round(opt.img_size * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            
            logits_high, sim_map, gate_weights = model(images)
            
            res = F.interpolate(logits_high, size=gts.shape[2:], mode='bilinear', align_corners=False)
            
            loss = structure_loss(res, gts)
            
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            
            if rate == 1:
                loss_record.update(loss.detach(), opt.batchsize)
                
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:0.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))

    # 儲存最新的模型狀態
    save_path = opt.train_save
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, 'latest.pth'))
    
    # Validation
    if epoch % 1 == 0:
        total_dataset_dice = 0
        total_dice = 0
        total_images = 0
        datasets = ['CVC-ClinicDB', 'Kvasir','CVC-300',  'CVC-ColonDB', 'ETIS-LaribPolypDB']
        
        print(f'\n[Validation] Epoch {epoch}')
        for dataset in datasets:
            dataset_dice, n_images = test(model, test_path, dataset, opt.img_size)
            total_dataset_dice += dataset_dice
            total_dice += (n_images * dataset_dice)
            total_images += n_images
            
            logging.info('epoch: {}, dataset: {}, dice: {:.4f}'.format(epoch, dataset, dataset_dice))
            print('{}: {:.4f}'.format(dataset, dataset_dice))
            dict_plot[dataset].append(dataset_dice)
            
        meandice = total_dice / total_images
        avg_dataset_dice = total_dataset_dice / len(datasets)
        dict_plot['test'].append(meandice)
        
        print('Mean Dice (Weighted by Image): {:.4f}'.format(meandice))
        print('Mean Dice (Average of Datasets): {:.4f}'.format(avg_dataset_dice))
        logging.info('Validation mean dice: {:.4f}'.format(avg_dataset_dice))
        
        if avg_dataset_dice > best:
            print('##################### Dice score improved from {:.4f} to {:.4f}'.format(best, avg_dataset_dice))
            logging.info('##################### Dice score improved from {:.4f} to {:.4f}'.format(best, avg_dataset_dice))
            best = avg_dataset_dice
            torch.save(model.state_dict(), os.path.join(save_path, 'best.pth'))
            
        print()
        model.train()

if __name__ == '__main__':
    dict_plot = {'CVC-300':[], 'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-ColonDB':[], 'ETIS-LaribPolypDB':[], 'test':[]}
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50)
    
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
    
    parser.add_argument('--name', type=str, default='adaptive_prototype_64')
    
    parser.add_argument('--backbone', type=str, default='vits16plus')
    
    parser.add_argument('--freeze', type=bool, default=False)
    
    parser.add_argument('--layer_indices', type=tuple, default=(1, 3, 5, 7, 9, 11))
    
    opt = parser.parse_args()
    
    logging.basicConfig(filename='train_log_'+opt.name+'.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    
    to_save = opt.backbone + '/' + opt.name
    opt.train_save = os.path.join(opt.train_save, to_save)
    
    torch.cuda.set_device(opt.device)
    
    model = DINOv3_Prototype_Adaptive(pool_size=64, backbone_type=opt.backbone, freeze=opt.freeze, layer_indices=opt.layer_indices).cuda()    
    
    best = 0.0
    
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(params, lr=opt.lr, weight_decay=1e-4)
    
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)
    
    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.img_size, augmentation=opt.augmentation)

    print("######## Start Training Adaptive Prototype ########")
    for epoch in range(1, opt.epoch + 1):
        # adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, opt.test_path, opt)