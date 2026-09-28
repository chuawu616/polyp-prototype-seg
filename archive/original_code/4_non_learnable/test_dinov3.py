import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc, ndimage
import cv2

from lib.networks_dinov3_concat import DINOv3_concat
from utils.dataloader import test_dataset
from tqdm import tqdm

if __name__ == '__main__':
    method_name = 'PolypPVT-CASCADE'
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default='/home/U116med/wch_code/non_learnable/models/polyp/vits16plus/baseline_dinov3/best.pth')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--mode', type=str, default='test')
    opt = parser.parse_args()
    
    #torch.cuda.set_device(0)  # set your gpu device
    
    model = DINOv3_concat(backbone_type='vits16plus')
    model.cuda()
    model.load_state_dict(torch.load(opt.pth_path))
    
    model.eval()
    if opt.mode=='train':
        root_path = '/home/U116med/data/polyp/'
        datasets = ['TrainDataset']
    else:
        root_path = '/home/U116med/data/polyp/TestDataset/'
        datasets = [ 'CVC-ClinicDB', 'Kvasir', 'CVC-300', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
    
    for _data_name in datasets:
        ##### put data_path here #####
        data_path = root_path + _data_name
        
        ##### save_path #####
        #save_path = './result_map/'+method_name+'/{}/'.format(_data_name)

        #if not os.path.exists(save_path):
        #    os.makedirs(save_path)
        
        print('Evaluating ' + data_path)
        
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, 352)
        DSC = 0.0
        JACARD = 0.0
        ACC_EDGE = 0.0
        ACC_AREA = 0.0
        for i in tqdm(range(num1)):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            
            res1 = model(image) # forward
            
            # eval Dice
            res = F.interpolate(res1, size=gt.shape, mode='bilinear', align_corners=False)

            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            #cv2.imwrite(save_path+name, res*255)        
            
            input = res#np.where(res >= 0.5, 1, 0) 
            target = np.where(np.array(gt) >= 0.5, 1, 0)
            
            # distance_map0 = ndimage.distance_transform_edt(target)
            # distance_map1 = ndimage.distance_transform_edt(1-target)
            # edge = 1*(distance_map0<=5)*target+1*(distance_map1<=5)*(1-target)
            # edge_flat = np.reshape(edge, (-1))
            
            smooth = 1
            input_flat = np.reshape(input, (-1))
            target_flat = np.reshape(target, (-1))
            intersection = (input_flat * target_flat)
            union = input_flat + target_flat - intersection
            
            jacard = ((np.sum(intersection)+smooth)/(np.sum(union)+smooth))
            jacard = '{:.4f}'.format(jacard)
            jacard = float(jacard)
            JACARD += jacard
            
            dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
            dice = '{:.4f}'.format(dice)
            dice = float(dice)
            DSC += dice
            
            # tp = 1*(input_flat==target_flat)
            # acc_edge = (tp*edge_flat).sum()/edge_flat.sum()
            # acc_area = (tp*(1-edge_flat)).sum()/(1-edge_flat).sum()
            # ACC_EDGE += acc_edge
            # ACC_AREA += acc_area
            
        print('*****************************************************')
        print('Dice Score: {:.3f}'.format(DSC/num1))
        print('Jacard Score: {:.3f}'.format(JACARD/num1))
        # print('Acc EDGE Score: {:.3f}'.format(ACC_EDGE/num1))
        # print('Acc AREA Score: {:.3f}'.format(ACC_AREA/num1))
        print(_data_name, 'Finish!')
        print('*****************************************************')
