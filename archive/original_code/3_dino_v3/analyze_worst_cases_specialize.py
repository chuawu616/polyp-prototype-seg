import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm

from lib.networks_prototype import DINOv3_Prototype
from lib.networks_prototype_concat import DINOv3_Prototype_concat
from lib.networks_prototype_specialize import DINOv3_Prototype_Specialize
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

def save_prototype_worst_case(original_image, true_mask, binary_pred, k_class_pred, k_class_conf, prob_easy_map, dice_score, output_path, fg_num, bg_num):
    """
    生成五聯圖：原圖 | 真值 | 二元預測 | K類子預測(漸層) | 門控權重熱力圖
    """
    h, w, _ = original_image.shape
    
    def _resize(m):
        if m.shape != (h, w):
            if m.dtype == np.float32 or m.dtype == np.float64:
                return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                return cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return m
    
    true_mask = _resize(true_mask)
    binary_pred = _resize(binary_pred)
    k_class_pred = _resize(k_class_pred)
    k_class_conf = _resize(k_class_conf) 
    prob_easy_map = _resize(prob_easy_map) # 縮放門控矩陣

    font = cv2.FONT_HERSHEY_SIMPLEX
    def add_title_and_border(img, txt, text_color=(0, 0, 0)):
        img = cv2.copyMakeBorder(img, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.putText(img, txt, (10, 30), font, 0.6, text_color, 2, cv2.LINE_AA)
        return img

    # [A] Original
    vis_original = add_title_and_border(original_image.copy(), "Original")

    # [B] Ground Truth
    vis_gt = original_image.copy()
    overlay = vis_gt.copy()
    overlay[true_mask == 1] = [0, 255, 255]
    vis_gt = cv2.addWeighted(overlay, 0.5, vis_gt, 0.5, 0)
    vis_gt = add_title_and_border(vis_gt, "Ground Truth")

    # [C] Binary Prediction
    vis_binary = original_image.copy()
    overlay = vis_binary.copy()
    overlay[binary_pred == 1] = [0, 255, 0]
    vis_binary = cv2.addWeighted(overlay, 0.5, vis_binary, 0.5, 0)
    title_text = f"Binary (DICE: {dice_score:.4f})"
    vis_binary = add_title_and_border(vis_binary, title_text, text_color=(0, 0, 255))
    
    # [D] K-Class Prototypes
    fg_palette = np.array([
    [0, 255, 0], [255, 100, 255],[0, 255, 255],[255, 255, 0],
    [255, 150, 0],[255, 50, 50],[100, 200, 255],[255, 255, 255],
    ], dtype=np.uint8)

    bg_palette = np.array([
    [0, 0, 80],[50, 0, 0],[0, 50, 0],[50, 50, 0],      
    [50, 0, 50],[0, 50, 50],[30, 30, 30],[0, 0, 0],        
    ], dtype=np.uint8)

    k_colored_map = np.zeros((h, w, 3), dtype=np.uint8)    
    for fg_id in range(min(len(fg_palette), fg_num)):
        k_colored_map[k_class_pred == fg_id] = fg_palette[fg_id]
    for bg_id in range(min(len(bg_palette), bg_num)):
        k_colored_map[k_class_pred == (fg_num + bg_id)] = bg_palette[bg_id]
    vis_k = add_title_and_border(k_colored_map, "K-Class Prototypes")

    # [E] Gate Visualization (prob_easy 熱力圖)
    # 將 0~1 的機率值轉換為 0~255 的灰階圖
    prob_easy_uint8 = np.clip(prob_easy_map * 255, 0, 255).astype(np.uint8)
    # 套用 JET 顏色映射：值越大(接近1)越紅，值越小(接近0)越藍
    vis_gate = cv2.applyColorMap(prob_easy_uint8, cv2.COLORMAP_JET)
    vis_gate = add_title_and_border(vis_gate, "Gate(Red:Easy, Blue:Hard)")

    # 拼接 5 張圖
    combined = np.concatenate((vis_original, vis_gt, vis_binary, vis_k, vis_gate), axis=1)
    cv2.imwrite(output_path, combined)

def main():
    parser = argparse.ArgumentParser(description="Find the worst 20 cases for Prototype Model")
    parser.add_argument('--pth_path', type=str, 
                        default= '/home/U116med/wch_code/cascade/models/prototype_cascade/baseline_prototype_fhb_424_warm_10_orth_loss/best.pth', 
                        help='訓練好的 prototype best.pth 路徑')
    
    parser.add_argument('--data_path', type=str, 
                        default='/home/U116med/data/polyp/TestDataset/test', 
                        help='測試資料集的路徑')
    
    parser.add_argument('--output_dir', type=str, 
                        default='./best_40_prototype_844_specialize_warm_10/', 
                        help='儲存資料夾')
    
    parser.add_argument('--backbone', type=str, 
                        default='vits16plus', 
                        help='使用的 DINOv3 版本')
    
    parser.add_argument('--testsize', type=int, default=352, help='推論尺寸')
    parser.add_argument('--num_cases', type=int, default=40, help='挑選數量')
    parser.add_argument('--num_fg', type=int, default=8, help='前景原型數量')
    parser.add_argument('--num_hard', type=int, default=4, help='困難原型數量')
    parser.add_argument('--num_bg', type=int, default=4, help='背景原型數量')
    opt = parser.parse_args()

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    print("正在載入 Prototype 模型...")
    model = DINOv3_Prototype_Specialize(num_fg=opt.num_fg, num_hard=opt.num_hard, num_bg=opt.num_bg, backbone_type=opt.backbone).cuda()
    
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

            # 推論取得二元分類與多類別相似度圖
            logits_binary, logits_k_class = model(image_tensor)
            
            # --- 重建 prob_easy 門控矩陣 ---
            # 根據模型設計，前 num_fg-1 個通道是 easy，接下來 1 個通道是 hard，最後 num_bg 個通道是 bg
            num_easy = opt.num_fg - 1
            sim_easy = logits_k_class[:, :num_easy, :, :]
            sim_bg = logits_k_class[:, opt.num_fg:, :, :]
            
            score_easy = torch.logsumexp(sim_easy, dim=1)
            score_bg = torch.logsumexp(sim_bg, dim=1)
            prob_easy = torch.sigmoid(score_easy - score_bg) # shape: (1, H, W)
            # -----------------------------
            
            # 二元預測處理
            res = (logits_binary[:, 1:2, :, :] - logits_binary[:, 0:1, :, :]) * 10.0
            res = F.interpolate(res, size=gt_mask.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            
            pred_binary = np.where(res >= 0.5, 1, 0)
            gt_binary = np.where(gt_mask >= 0.5, 1, 0)
            
            # K-Class 原型特徵圖處理
            sim_map_resized = F.interpolate(logits_k_class, size=gt_mask.shape, mode='bilinear', align_corners=False)
            sim_map_squeezed = sim_map_resized.squeeze(0).cpu().numpy()
            
            pred_k_mask = sim_map_squeezed.argmax(axis=0)
            pred_k_conf = sim_map_squeezed.max(axis=0)
            
            # 門控權重圖處理 (調整尺寸與轉為 numpy)
            prob_easy_resized = F.interpolate(prob_easy.unsqueeze(1), size=gt_mask.shape, mode='bilinear', align_corners=False)
            prob_easy_np = prob_easy_resized.squeeze().cpu().numpy()
            
            smooth = 1
            intersection = (pred_binary.reshape(-1) * gt_binary.reshape(-1)).sum()
            dice = (2 * intersection + smooth) / (pred_binary.sum() + gt_binary.sum() + smooth)
            
            results_list.append({
                'name': name,
                'dice': float(dice),
                'pred_mask': pred_binary,
                'gt_mask': gt_binary,
                'k_mask': pred_k_mask,
                'k_conf': pred_k_conf,
                'prob_easy': prob_easy_np, # 紀錄門控矩陣
                'orig_img': tensor_to_bgr_image(image_tensor)
            })

    results_list.sort(key=lambda x: x['dice'])
    # worst_cases = results_list[:opt.num_cases]
    worst_cases = results_list[-opt.num_cases:]
    
    print(f"\n生成 DICE 表現最低的 {opt.num_cases} 筆資料比對圖...")
    
    for rank, case in enumerate(worst_cases, start=1):
        save_name = f"Rank{rank:02d}_DICE_{case['dice']:.4f}_{case['name']}"
        save_path = os.path.join(opt.output_dir, save_name)
        
        save_prototype_worst_case(
            original_image=case['orig_img'],
            true_mask=case['gt_mask'],
            binary_pred=case['pred_mask'],
            k_class_pred=case['k_mask'],
            k_class_conf=case['k_conf'],
            prob_easy_map=case['prob_easy'], # 傳入門控矩陣
            dice_score=case['dice'],
            output_path=save_path,
            fg_num=opt.num_fg,
            bg_num=opt.num_bg
        )

    print(f"\n完成！圖片已儲存至 {opt.output_dir}")

if __name__ == '__main__':
    main()