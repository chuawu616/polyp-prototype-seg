import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm

from lib.networks_prototype_specialize import DINOv3_Prototype_Specialize
from lib.networks_prototype_specialize_pvtv2 import Prototype_CASCADE_Specialize
from lib.networks_prototype_concat import DINOv3_Prototype_concat
from lib.networks_prototype import DINOv3_Prototype
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

def save_divergent_comparison(img_bgr, gt_mask, pred_a, pred_b, k_map_a, k_map_b, dice_a, dice_b, diff, save_path, fg_num_a, bg_num_a, fg_num_b, bg_num_b):
    """
    生成五聯圖：Original | GT | Proto A Pred | Proto B Pred | Proto B K-Class
    """
    h, w = gt_mask.shape[:2]
    
    if img_bgr.shape[:2] != (h, w):
        img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    def add_title_and_border(img, txt, text_color=(0, 0, 0)):
        img = cv2.copyMakeBorder(img, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.putText(img, txt, (10, 30), font, 0.6, text_color, 2, cv2.LINE_AA)
        return img

    # [1] Original
    vis_original = add_title_and_border(img_bgr.copy(), f"Diff: {diff:.4f}")

    # [2] GT (黃色)
    vis_gt = img_bgr.copy()
    overlay = vis_gt.copy()
    overlay[gt_mask == 1] = [0, 255, 255]
    vis_gt = cv2.addWeighted(overlay, 0.5, vis_gt, 0.5, 0)
    vis_gt = add_title_and_border(vis_gt, "Ground Truth")

    # [3] Proto A Pred (綠色)
    vis_a = img_bgr.copy()
    overlay = vis_a.copy()
    overlay[pred_a == 1] = [0, 255, 0]
    vis_a = cv2.addWeighted(overlay, 0.5, vis_a, 0.5, 0)
    vis_a = add_title_and_border(vis_a, f"Proto A DICE: {dice_a:.4f}", text_color=(255, 0, 0))

    # [4] Proto B Pred (綠色)
    vis_b = img_bgr.copy()
    overlay = vis_b.copy()
    overlay[pred_b == 1] = [0, 255, 0]
    vis_b = cv2.addWeighted(overlay, 0.5, vis_b, 0.5, 0)
    vis_b = add_title_and_border(vis_b, f"Proto B DICE: {dice_b:.4f}", text_color=(0, 0, 255))

    # [5] Prototype A K-Class
    fg_palette = np.array([
    [0, 255, 0], [255, 100, 255],[0, 255, 255],[255, 255, 0],
    [255, 150, 0],[255, 50, 50],[100, 200, 255],[255, 255, 255],
    ], dtype=np.uint8)

    bg_palette = np.array([
    [0, 0, 80],[50, 0, 0],[0, 50, 0],[50, 50, 0],      
    [50, 0, 50],[0, 50, 50],[30, 30, 30],[0, 0, 0],        
    ], dtype=np.uint8)

    k_colored_map_a = np.zeros((h, w, 3), dtype=np.uint8)    
    for fg_id in range(min(len(fg_palette), fg_num_a)):
        k_colored_map_a[k_map_a == fg_id] = fg_palette[fg_id]
    for bg_id in range(min(len(bg_palette), bg_num_a)):
        k_colored_map_a[k_map_a == (fg_num_a + bg_id)] = bg_palette[bg_id]
        
    vis_k_a = add_title_and_border(k_colored_map_a, "Proto A K-Class")
    
    # [5] Prototype B K-Class
    k_colored_map_b = np.zeros((h, w, 3), dtype=np.uint8)    
    for fg_id in range(min(len(fg_palette), fg_num_b)):
        k_colored_map_b[k_map_b == fg_id] = fg_palette[fg_id]
    for bg_id in range(min(len(bg_palette), bg_num_b)):
        k_colored_map_b[k_map_b == (fg_num_b + bg_id)] = bg_palette[bg_id]
        
    vis_k_b = add_title_and_border(k_colored_map_b, "Proto B K-Class")
    combined = np.concatenate((vis_original, vis_gt, vis_a, vis_b, vis_k_a, vis_k_b), axis=1)
    cv2.imwrite(save_path, combined)

def load_weights_robustly(model, pth_path):
    checkpoint = torch.load(pth_path, map_location='cuda')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)
    return model

def main():
    parser = argparse.ArgumentParser(description="Compare two Prototype models")
    parser.add_argument('--proto_a_pth', type=str, 
                        default='/home/U116med/wch_code/dino_v3/models/prototype_cascade/vits16plus/baseline_prototype_fb_44_concat_fine/best.pth', 
                        help='第一個模型權重路徑')
    parser.add_argument('--proto_b_pth', type=str, 
                        default='/home/U116med/wch_code/dino_v3/models/prototype_cascade/vits16plus/baseline_prototype_fhb_424_specialize_warm_10/best.pth', 
                        help='第二個模型權重路徑')
    # '/home/U116med/wch_code/cascade/models/prototype_cascade/baseline_prototype_fhb_424_warm_10/best.pth'
    
    parser.add_argument('--data_path', type=str, default='/home/U116med/data/polyp/TestDataset/test/')
    
    parser.add_argument('--output_dir', type=str, default='./dinov3_specialize_or_not_fb44_comparison/')
    
    parser.add_argument('--backbone', type=str, default='vits16plus')
    # 假設兩個模型可能使用不同的 Prototype 數量
    parser.add_argument('--num_fg_a', type=int, default=4)

    parser.add_argument('--num_hard_a', type=int, default=2)

    parser.add_argument('--num_bg_a', type=int, default=4)
    
    parser.add_argument('--num_fg_b', type=int, default=4)

    parser.add_argument('--num_hard_b', type=int, default=2)
        
    parser.add_argument('--num_bg_b', type=int, default=4)
    
    parser.add_argument('--testsize', type=int, default=352)
    
    parser.add_argument('--num_cases', type=int, default=35)
    
    opt = parser.parse_args()

    os.makedirs(opt.output_dir, exist_ok=True)

    print("載入 Prototype A...")
    model_a = DINOv3_Prototype_concat(num_fg=opt.num_fg_a, num_bg=opt.num_bg_a, backbone_type=opt.backbone).cuda()
    # model_a = DINOv3_Prototype_Specialize(num_fg=opt.num_fg_a, num_hard=opt.num_hard_a, num_bg=opt.num_bg_a, backbone_type=opt.backbone).cuda()
    model_a = load_weights_robustly(model_a, opt.proto_a_pth)
    model_a.eval()

    print("載入 Prototype B...")
    # model_b = DINOv3_Prototype_concat(num_fg=opt.num_fg_b, num_bg=opt.num_bg_b, backbone_type=opt.backbone).cuda()
    # model_b = Prototype_CASCADE_Specialize(num_fg=opt.num_fg_b, num_hard=opt.num_hard_b, num_bg=opt.num_bg_b).cuda()
    model_b = DINOv3_Prototype_Specialize(num_fg=opt.num_fg_b, num_hard=opt.num_hard_b, num_bg=opt.num_bg_b, backbone_type=opt.backbone).cuda()
    model_b = load_weights_robustly(model_b, opt.proto_b_pth)
    model_b.eval()

    test_loader = test_dataset(os.path.join(opt.data_path, 'images/'), 
                               os.path.join(opt.data_path, 'masks/'), 
                               opt.testsize)
    num_images = test_loader.size
    results_list = []

    print("執行雙模型推論...")
    with torch.no_grad():
        for i in tqdm(range(num_images)):
            image_tensor, gt_mask, name = test_loader.load_data()
            gt_mask = np.asarray(gt_mask, np.float32)
            gt_mask /= (gt_mask.max() + 1e-8)
            image_tensor = image_tensor.cuda()
            gt_binary = np.where(gt_mask >= 0.5, 1, 0)
            target_flat = gt_binary.reshape(-1)
            smooth = 1.0

            # 推論 Model A
            logits_a, sim_map_a = model_a(image_tensor)
            res_a = (logits_a[:, 1:2, :, :] - logits_a[:, 0:1, :, :]) * 10.0
            res_a = F.interpolate(res_a, size=gt_mask.shape, mode='bilinear', align_corners=False)
            res_a = res_a.sigmoid().data.cpu().numpy().squeeze()
            res_a = (res_a - res_a.min()) / (res_a.max() - res_a.min() + 1e-8)
            pred_a = np.where(res_a >= 0.5, 1, 0)
            dice_a = (2 * (pred_a.reshape(-1) * target_flat).sum() + smooth) / (pred_a.sum() + gt_binary.sum() + smooth)

            # 取得 Model A 的 K-Map
            sim_map_a_resized = F.interpolate(sim_map_a, size=gt_mask.shape, mode='bilinear', align_corners=False)
            k_map_a = sim_map_a_resized.argmax(dim=1).squeeze(0).cpu().numpy()

            # 推論 Model B
            logits_b, sim_map_b = model_b(image_tensor)
            res_b = (logits_b[:, 1:2, :, :] - logits_b[:, 0:1, :, :]) * 10.0
            res_b = F.interpolate(res_b, size=gt_mask.shape, mode='bilinear', align_corners=False)
            res_b = res_b.sigmoid().data.cpu().numpy().squeeze()
            res_b = (res_b - res_b.min()) / (res_b.max() - res_b.min() + 1e-8)
            pred_b = np.where(res_b >= 0.5, 1, 0)
            dice_b = (2 * (pred_b.reshape(-1) * target_flat).sum() + smooth) / (pred_b.sum() + gt_binary.sum() + smooth)

            # 取得 Model B 的 K-Map
            sim_map_b_resized = F.interpolate(sim_map_b, size=gt_mask.shape, mode='bilinear', align_corners=False)
            k_map_b = sim_map_b_resized.argmax(dim=1).squeeze(0).cpu().numpy()

            diff = abs(dice_a - dice_b)
            results_list.append({
                'name': name,
                'dice_a': float(dice_a),
                'dice_b': float(dice_b),
                'diff': float(diff),
                'gt_mask': gt_binary,
                'pred_a': pred_a,
                'pred_b': pred_b,
                'k_map_a': k_map_a,
                'k_map_b': k_map_b,
                'orig_img': tensor_to_bgr_image(image_tensor)
            })

    results_list.sort(key=lambda x: x['diff'], reverse=True)
    top_divergent_cases = results_list[:opt.num_cases]
    
    for rank, case in enumerate(top_divergent_cases, start=1):
        winner = "ProtoA" if case['dice_a'] > case['dice_b'] else "ProtoB"
        save_name = f"Rank{rank:02d}_Diff{case['diff']:.4f}_{winner}Win_{case['name']}"
        save_path = os.path.join(opt.output_dir, save_name)
        
        save_divergent_comparison(
            img_bgr=case['orig_img'],
            gt_mask=case['gt_mask'],
            pred_a=case['pred_a'],
            pred_b=case['pred_b'],
            k_map_a=case['k_map_a'],
            k_map_b=case['k_map_b'],
            dice_a=case['dice_a'],
            dice_b=case['dice_b'],
            diff=case['diff'],
            save_path=save_path,
            fg_num_a=opt.num_fg_a,
            bg_num_a=opt.num_bg_a,
            fg_num_b=opt.num_fg_b,
            bg_num_b=opt.num_bg_b
        )

    print(f"\n分析完成，結果儲存於：{opt.output_dir}")

if __name__ == '__main__':
    main()