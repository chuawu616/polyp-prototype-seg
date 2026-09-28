import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from tqdm import tqdm
from lib.networks_prototype import DINOv3_Prototype
from lib.networks_prototype_concat import DINOv3_Prototype_concat
from lib.networks_prototype_specialize import DINOv3_Prototype_Specialize

from lib.networks_prototype_specialize_pvtv2 import Prototype_CASCADE_Specialize
from lib.networks_prototype_hybrid_specialize_pvtv2 import Prototype_CASCADE_Hybrid_Specialize
from utils.dataloader import test_dataset
import sys

def print_similarity_matrix(model, num_fg, num_bg):
    if hasattr(model, 'prototypes'):
        prototypes = model.prototypes.data.cpu().numpy()
    elif hasattr(model, 'prototype_head') and hasattr(model.prototype_head, 'prototypes'):
        prototypes = model.prototype_head.prototypes.data.cpu().numpy()
    else:
        return

    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    prototypes_norm = prototypes / (norms + 1e-8)
    sim_matrix = np.dot(prototypes_norm, prototypes_norm.T)

    labels = [f"F{i+1}" for i in range(num_fg)] + [f"B{i+1}" for i in range(num_bg)]
    
    print("\n" + "="*60)
    print("        Learned Prototypes Cosine Similarity Matrix")
    print("="*60)

    header = "      " + " ".join([f"{l:>6}" for l in labels])
    print(header)
    print("-" * len(header))

    for i, row in enumerate(sim_matrix):
        row_vals = " ".join([f"{val:>6.2f}" for val in row])
        print(f"{labels[i]:>5} | {row_vals}")
    
    print("="*60 + "\n")

def print_similarity_matrix_specialize(model):
    # 1. 檢查並提取 Prototypes
    if hasattr(model, 'easy_prototypes'):
        p_easy = model.easy_prototypes.data.cpu().numpy() # [num_easy, dim]
        p_hard = model.hard_prototypes.data.cpu().numpy() # [num_hard, dim]
        p_bg = model.bg_prototypes.data.cpu().numpy()     # [num_bg, dim]
    else:
        return

    # 2. 拼接所有 Prototypes 並進行正規化
    # 順序：Easy -> Hard -> Background
    prototypes = np.concatenate([p_easy, p_hard, p_bg], axis=0)
    
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    prototypes_norm = prototypes / (norms + 1e-8)
    
    # 3. 計算餘弦相似度 (Dot Product of Normalized Vectors)
    sim_matrix = np.dot(prototypes_norm, prototypes_norm.T)

    # 4. 準備標籤
    num_easy = p_easy.shape[0]
    num_hard = p_hard.shape[0]
    num_bg = p_bg.shape[0]
    
    labels = ([f"E{i+1}" for i in range(num_easy)] + 
              [f"H{i+1}" for i in range(num_hard)] + 
              [f"B{i+1}" for i in range(num_bg)])
    
    # 5. 格式化輸出
    print("\n" + "="*75)
    print("      Learned Prototypes Cosine Similarity Matrix (E/H/B)")
    print("="*75)

    # 標頭 (Header)
    header = "      " + " ".join([f"{l:>7}" for l in labels])
    print(header)
    print("-" * len(header))

    # 矩陣內容
    for i, row in enumerate(sim_matrix):
        row_vals = " ".join([f"{val:>7.2f}" for val in row])
        print(f"{labels[i]:>5} | {row_vals}")
    
    print("-" * len(header))
    print("="*75 + "\n")

def evaluate_datasets(model, root_path, dataset_list, testsize):
    for group_name, datasets in dataset_list.items():
        print(f"\n===== Evaluating {group_name} datasets =====")
        total_DSC = 0.0
        total_JACARD = 0.0
        total_images = 0

        for _data_name in datasets:
            data_path = os.path.join(root_path, _data_name)
            image_root = os.path.join(data_path, 'images/')
            gt_root = os.path.join(data_path, 'masks/')

            if not os.path.exists(image_root):
                print(f"[Skip] {_data_name} not found")
                continue

            test_loader = test_dataset(image_root, gt_root, testsize)
            num_images = test_loader.size
            total_images += num_images

            DSC = 0.0
            JACARD = 0.0
            
            print(f'Evaluating {_data_name} ({num_images} images) ...')

            for i in tqdm(range(num_images)):
                image, gt, name = test_loader.load_data()
                
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                image = image.cuda()

                with torch.no_grad():
                    res, _ = model(image)
                
                # Prototype Inference Logic
                res = (res[:, 1:2, :, :] - res[:, 0:1, :, :]) * 10.0
                res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)

                input = res #np.where(res >= 0.5, 1, 0)
                target = np.where(gt >= 0.5, 1, 0)
                
                input_flat = np.reshape(input, (-1))
                target_flat = np.reshape(target, (-1))
                
                intersection = (input_flat * target_flat)
                union = input_flat + target_flat - intersection
                
                smooth = 1
                
                # Jaccard
                jacard = (np.sum(intersection) + smooth) / (np.sum(union) + smooth)
                JACARD += jacard
                
                # Dice
                dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
                DSC += dice

            print(f'{_data_name} Results: Mean Dice={DSC/num_images:.4f}, Mean Jaccard={JACARD/num_images:.4f}\n')
            total_DSC += DSC
            total_JACARD += JACARD

        if total_images > 0:
            print(f"--- {group_name} Average ---")
            print(f"Dice:    {total_DSC/total_images:.4f}")
            print(f"Jaccard: {total_JACARD/total_images:.4f}")
            print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, 
                        default="/home/U116med/wch_code/cascade/models/prototype_cascade/baseline_prototype_hybrid_fhb_1084_extract_warm/best.pth", 
                        help='Path to best_model.pth')
    parser.add_argument('--num_fg', type=int, default=10)
    parser.add_argument('--num_hard', type=int, default=8)
    parser.add_argument('--num_bg', type=int, default=4)
    opt = parser.parse_args()

    # model = DINOv3_Prototype(num_fg=opt.num_fg, num_bg=opt.num_bg, backbone_type='vits16plus').cuda()
    # model = DINOv3_Prototype_concat(num_fg=opt.num_fg, num_bg=opt.num_bg, backbone_type='vits16plus').cuda()
    # model = DINOv3_Prototype_Specialize(num_fg=opt.num_fg, num_hard=opt.num_hard, num_bg=opt.num_bg, backbone_type='vits16plus').cuda()
    # model = Prototype_CASCADE_Specialize(num_fg=opt.num_fg, num_hard=opt.num_hard, num_bg=opt.num_bg).cuda()
    model = Prototype_CASCADE_Hybrid_Specialize(num_fg=opt.num_fg, num_hard=opt.num_hard, num_bg=opt.num_bg).cuda()
    
    if os.path.exists(opt.pth_path):
        print(f"Loading weights from: {opt.pth_path}")
        checkpoint = torch.load(opt.pth_path, map_location='cuda', weights_only=True)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     name = k.replace('module.', '')
        #     new_state_dict[name] = v
        new_state_dict = {}
        for k, v in state_dict.items():
            # 1. 移除 DataParallel 可能產生的 module. 前綴
            name = k.replace('module.', '')
        
            # 2. 修正 Hybrid 模組的名稱映射
            # 將 "hard_prototypes_XXX" 替換為 "hard_prototypes_module.XXX"
            if name.startswith('hard_prototypes_') and 'module' not in name:
                name = name.replace('hard_prototypes_', 'hard_prototypes_module.', 1)
            
            new_state_dict[name] = v
        
        try:
            model.load_state_dict(new_state_dict, strict=True)
            print("Weights loaded successfully (Strict Mode).")
        except RuntimeError:
            print("Strict loading failed. Loading with strict=False.")
            model.load_state_dict(new_state_dict, strict=False)
    else:
        print(f"Error: Path {opt.pth_path} not found.")
        exit()

    model.eval()

    root_path = '/home/U116med/data/polyp/TestDataset/'
    dataset_groups = {
        'In-Domain': ['CVC-ClinicDB', 'Kvasir'],
        'Out-of-Domain': ['CVC-300', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
    }

    evaluate_datasets(model, root_path, dataset_groups, opt.testsize)
    # print_similarity_matrix(model, opt.num_fg, opt.num_bg)
    # print_similarity_matrix_specialize(model)