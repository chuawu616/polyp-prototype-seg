import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm

from lib.networks_prototype_v2 import Prototype_CASCADE_v2
from lib.networks_prototype_v3 import Prototype_CASCADE_v3
from lib.networks_prototype_dinov3_v2 import Prototype_DINOv3_v2
from utils.dataloader import test_dataset

def tensor_to_bgr_image(tensor):
    """將 Normalize 過的 Tensor 還原回 BGR 圖片"""
    tensor = tensor.squeeze(0).cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    img = tensor * std + mean
    img = torch.clamp(img, 0, 1)
    img_np = img.permute(1, 2, 0).numpy()
    
    img_bgr = (img_np * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
    return img_bgr

def get_prototype_worst_case_img(original_image, true_mask, binary_pred, k_class_pred, dice_score, fg_num, bg_num):
    """
    生成四聯圖：原圖 | 真值 | 二元預測(帶分數) | K類子預測，並回傳圖片陣列
    """
    h, w, _ = original_image.shape
    
    def _resize(m):
        if m.shape != (h, w):
            return cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return m
    
    true_mask = _resize(true_mask)
    binary_pred = _resize(binary_pred)
    k_class_pred = _resize(k_class_pred)

    font = cv2.FONT_HERSHEY_SIMPLEX
    def add_title_and_border(img, txt, text_color=(0, 0, 0)):
        img = cv2.copyMakeBorder(img, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.putText(img, txt, (10, 30), font, 0.7, text_color, 2, cv2.LINE_AA)
        return img

    # [A] Original
    vis_original = add_title_and_border(original_image.copy(), "Original")

    # [B] Ground Truth (黃色 = FG)
    vis_gt = original_image.copy()
    overlay = vis_gt.copy()
    overlay[true_mask == 1] = [0, 255, 255]
    vis_gt = cv2.addWeighted(overlay, 0.5, vis_gt, 0.5, 0)
    vis_gt = add_title_and_border(vis_gt, "Ground Truth")

    # [C] Binary Prediction (綠色 = Pred) + 顯示 DICE 分數
    vis_binary = original_image.copy()
    overlay = vis_binary.copy()
    overlay[binary_pred == 1] = [0, 255, 0]
    vis_binary = cv2.addWeighted(overlay, 0.5, vis_binary, 0.5, 0)
    title_text = f"Binary (DICE: {dice_score:.4f})"
    vis_binary = add_title_and_border(vis_binary, title_text, text_color=(0, 0, 255))
    
    # [D] K-Class Prototypes
    bg_palette = np.array([
    [0, 255, 0], [255, 100, 255], [0, 255, 255], [255, 255, 0],   
    [255, 150, 0], [255, 50, 50], [100, 200, 255], [255, 255, 255], 
    [150, 255, 0], [0, 255, 150], [200, 100, 255], [255, 0, 150],   
    [100, 255, 200], [255, 200, 100], [50, 150, 255], [200, 255, 100], 
    [180, 180, 255], [255, 180, 180], [180, 255, 180], [220, 220, 220]  
    ], dtype=np.uint8)

    fg_palette = np.array([
    [0, 0, 80], [50, 0, 0], [0, 50, 0], [50, 50, 0],                
    [50, 0, 50], [0, 50, 50], [30, 30, 30], [0, 0, 0],              
    [0, 0, 50], [30, 0, 0], [0, 30, 0], [40, 40, 40],                
    [20, 20, 60], [60, 20, 20], [20, 60, 20], [60, 60, 0],            
    [40, 0, 80], [80, 0, 40], [0, 80, 40], [20, 20, 20]              
    ], dtype=np.uint8)

    k_colored_map = np.zeros((h, w, 3), dtype=np.uint8)    
    for fg_id in range(min(len(fg_palette), fg_num)):
        k_colored_map[k_class_pred == fg_id] = fg_palette[fg_id]
    for bg_id in range(min(len(bg_palette), bg_num)):
        k_colored_map[k_class_pred == (fg_num + bg_id)] = bg_palette[bg_id]
    vis_k = add_title_and_border(k_colored_map, "K-Class Prototypes")

    # 左右拼接成一張四聯圖
    combined = np.concatenate((vis_original, vis_gt, vis_binary, vis_k), axis=1)
    return combined

def main():
    parser = argparse.ArgumentParser(description="Find the worst cases for Prototype Model and concat them")
    parser.add_argument('--pth_path', type=str, 
                        default= '/home/U116med/wch_code/non_learnable/models/prototype_cascade/prototypev3_fb55_aug_str_replace_ce/best.pth', 
                        help='訓練好的 prototype best.pth 路徑')
    
    parser.add_argument('--data_path', type=str, 
                        default='/home/U116med/data/polyp/TestDataset/test'
                        , help='測試資料集的路徑')
    
    parser.add_argument('--output_dir', type=str, 
                        default='./worst_40_cases_fb55_prototypev3_aug_str_replace_ce/', 
                        help='儲存資料夾')
    
    parser.add_argument('--testsize', type=int, default=352, help='推論尺寸')
    parser.add_argument('--num_cases', type=int, default=40, help='挑選數量')
    parser.add_argument('--num_fg', type=int, default=5, help='前景原型數量')
    parser.add_argument('--num_bg', type=int, default=5, help='背景原型數量')
    opt = parser.parse_args()

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    print("正在載入 Prototype 模型...")
    model = Prototype_CASCADE_v3(num_classes=2, num_prototype=opt.num_fg).cuda()
    # model = Prototype_CASCADE_v2(num_classes=2, num_prototype=opt.num_fg).cuda()
    # model = Prototype_DINOv3_v2(num_classes=2, num_prototype=opt.num_fg).cuda()
    
    # 載入權重處理 (相容 module. 前綴)
    checkpoint = torch.load(opt.pth_path, map_location='cuda')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    image_root = os.path.join(opt.data_path, 'images/')
    gt_root = os.path.join(opt.data_path, 'masks/')
    
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    num_images = test_loader.size
    results_list = []

    print(f"開始進行全資料集推論 ({num_images} 張圖片)...")
    with torch.no_grad():
        for i in tqdm(range(num_images)):
            image_tensor, gt_mask, name = test_loader.load_data()
            
            gt_mask = np.asarray(gt_mask, np.float32)
            gt_mask /= (gt_mask.max() + 1e-8)
            image_tensor = image_tensor.cuda()

            # Prototype 模型推論
            logits_binary, logits_k_class = model(image_tensor)
            
            # 二元預測處理
            res = (logits_binary[:, 1:2, :, :] - logits_binary[:, 0:1, :, :]) * 10.0
            res = F.interpolate(res, size=gt_mask.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            
            pred_binary = np.where(res >= 0.5, 1, 0)
            gt_binary = np.where(gt_mask >= 0.5, 1, 0)
            
            # K-Class 原型特徵圖處理
            sim_map_resized = F.interpolate(logits_k_class, size=gt_mask.shape, mode='bilinear', align_corners=False)
            pred_k_mask = sim_map_resized.argmax(dim=1).squeeze(0).cpu().numpy()
            
            # 計算 DICE
            smooth = 1
            intersection = (pred_binary.reshape(-1) * gt_binary.reshape(-1)).sum()
            dice = (2 * intersection + smooth) / (pred_binary.sum() + gt_binary.sum() + smooth)
            
            results_list.append({
                'name': name,
                'dice': float(dice),
                'pred_mask': pred_binary,
                'gt_mask': gt_binary,
                'k_mask': pred_k_mask,
                'orig_img': tensor_to_bgr_image(image_tensor)
            })

    # 依照 DICE 升冪排序
    results_list.sort(key=lambda x: x['dice'])
    worst_cases = results_list[:opt.num_cases]
    # worst_cases = results_list[-opt.num_cases:]
    
    print(f"\n生成 DICE 表現最低的 {opt.num_cases} 筆資料拼接圖...")
    
    all_blocks = []
    for rank, case in enumerate(worst_cases, start=1):
        # 取得該 case 的水平四聯圖
        row_img = get_prototype_worst_case_img(
            original_image=case['orig_img'],
            true_mask=case['gt_mask'],
            binary_pred=case['pred_mask'],
            k_class_pred=case['k_mask'],
            dice_score=case['dice'],
            fg_num=opt.num_fg,
            bg_num=opt.num_bg
        )
        
        # 建立文字橫幅 (Banner)
        info_text = f"Rank: {rank:02d} | DICE: {case['dice']:.4f} | File: {case['name']}"
        banner_height = 60
        banner_width = row_img.shape[1]
        banner = np.ones((banner_height, banner_width, 3), dtype=np.uint8) * 255
        
        # 計算文字大小以達成置中效果
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(info_text, font, 1.0, 2)[0]
        text_x = (banner_width - text_size[0]) // 2
        text_y = (banner_height + text_size[1]) // 2
        
        # 繪製文字
        cv2.putText(banner, info_text, (text_x, text_y), font, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
        
        # 將橫幅與四聯圖上下拼接為一個區塊
        case_block = np.concatenate((banner, row_img), axis=0)
        all_blocks.append(case_block)

    # 將所有區塊上下拼接成最終的一張大圖
    final_combined_image = np.concatenate(all_blocks, axis=0)
    
    save_path = os.path.join(opt.output_dir, f"top_{opt.num_cases}_worst_cases_combined.png")
    # save_path = os.path.join(opt.output_dir, f"top_{opt.num_cases}_best_cases_combined.png")
    cv2.imwrite(save_path, final_combined_image)

    print(f"\n完成！圖片已合併並儲存至 {save_path}")

if __name__ == '__main__':
    main()