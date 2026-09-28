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
from lib.networks_prototype_specialize import Prototype_CASCADE_Specialize
from lib.networks_prototype_hybrid_specialize import Prototype_CASCADE_Hybrid_Specialize
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter

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

        # Forward
        # 我們的模型返回 (logits, similarity_map)
        res, _ = model(image)
        
        # 取 (FG - BG) * 10
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

def train(train_loader, model, optimizer, epoch, test_path, warmup_epoch):
    model.train()
    global best
    size_rates = [0.75, 1, 1.25] 
    loss_record = AvgMeter()

    current_alpha = min(1.0, epoch / warmup_epoch)
    
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
            
            # logits, sim_map = model(images)
            logits, sim_map = model(images, alpha=current_alpha)
            
            # 確保 logits 與 gts 尺寸一致
            if logits.shape[-2:] != gts.shape[-2:]:
                logits = F.interpolate(logits, size=gts.shape[-2:], mode='bilinear', align_corners=False)
            
            final_pred = (logits[:, 1:2, :, :] - logits[:, 0:1, :, :]) * 10.0
            
            loss = structure_loss(final_pred, gts)

            # orth_loss = orthogonality_loss(model.easy_prototypes, model.hard_prototypes)
            
            # loss += orth_loss * 0.1
            
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)
                
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:0.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))
            # print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:0.4f}, orth_loss: {:0.4f}'.
                  # format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show(), orth_loss))

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
    
    parser.add_argument('--device', type=int, default=1)

    parser.add_argument('--name', type=str, default='baseline_prototype_hybrid_fhb_1084_extract_warm')

    parser.add_argument('--fg_num', type=int, default=10)

    parser.add_argument('--hard_num', type=int, default=8)

    parser.add_argument('--bg_num', type=int, default=4)

    parser.add_argument('--warmup_epoch', type=int, default=10)
    
    opt = parser.parse_args()
    logging.basicConfig(filename='train_log_'+opt.name+'.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    opt.train_save = os.path.join(opt.train_save, opt.name)
    
    torch.cuda.set_device(opt.device)
    
    # model = Prototype_CASCADE(num_fg=opt.fg_num, num_bg=opt.bg_num, encoder_path=None).cuda()
    # model = Prototype_CASCADE_Specialize(num_fg=opt.fg_num, num_hard=opt.hard_num, num_bg=opt.bg_num, encoder_path=None).cuda()
    model = Prototype_CASCADE_Hybrid_Specialize(num_fg=opt.fg_num, num_hard=opt.hard_num, num_bg=opt.bg_num, encoder_path=None).cuda()
    
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
        train(train_loader, model, optimizer, epoch, opt.test_path, opt.warmup_epoch)