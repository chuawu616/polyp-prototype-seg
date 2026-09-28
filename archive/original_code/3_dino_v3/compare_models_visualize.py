import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm

from lib.networks_concat import DINOv3_MLP_concat
from lib.networks import DINOv3_MLP
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

def save_divergent_comparison(img_bgr, gt_mask, mlp_pred, proto_pred, proto_k_map, mlp_dice, proto_dice, diff, save_path, fg_num, bg_num):
    """
    生成五聯圖：Original | GT | MLP Pred | Prototype Pred | Prototype K-Class
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

    # [3] MLP Pred (綠色)
    vis_mlp = img_bgr.copy()
    overlay = vis_mlp.copy()
    overlay[mlp_pred == 1] = [0, 255, 0]
    vis_mlp = cv2.addWeighted(overlay, 0.5, vis_mlp, 0.5, 0)
    vis_mlp = add_title_and_border(vis_mlp, f"MLP DICE: {mlp_dice:.4f}", text_color=(255, 0, 0))

    # [4] Prototype Pred (綠色)
    vis_proto = img_bgr.copy()
    overlay = vis_proto.copy()
    overlay[proto_pred == 1] = [0, 255, 0]
    vis_proto = cv2.addWeighted(overlay, 0.5, vis_proto, 0.5, 0)
    vis_proto = add_title_and_border(vis_proto, f"Proto DICE: {proto_dice:.4f}", text_color=(0, 0, 255))

    # [5] Prototype K-Class
    fg_palette = np.array([
        [0, 0, 255], [0, 165, 255], [0, 255, 255], [255, 0, 255],
        [0, 255, 128], [255, 0, 128], [128, 0, 255], [0, 64, 255]
    ], dtype=np.uint8)

    bg_palette = np.array([
        [255, 0, 0], [255, 255, 0], [128, 30, 170], [0, 130, 100],
        [128, 128, 128], [64, 64, 64], [128, 64, 0], [0, 128, 128]
    ], dtype=np.uint8)

    k_colored_map = np.zeros((h, w, 3), dtype=np.uint8)    
    for fg_id in range(min(len(fg_palette), fg_num)):
        k_colored_map[proto_k_map == fg_id] = fg_palette[fg_id]
    for bg_id in range(min(len(bg_palette), bg_num)):
        k_colored_map[proto_k_map == (fg_num + bg_id)] = bg_palette[bg_id]
        
    vis_k = add_title_and_border(k_colored_map, "Proto K-Class")

    # 橫向拼接五張圖片
    combined = np.concatenate((vis_original, vis_gt, vis_mlp, vis_proto, vis_k), axis=1)
    cv2.imwrite(save_path, combined)

def load_weights_robustly(model, pth_path):
    """相容帶有 'module.' 前綴的權重檔"""
    checkpoint = torch.load(pth_path, map_location='cuda')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)
    return model

def main():
    parser = argparse.ArgumentParser(description="Compare MLP and Prototype models to find largest discrepancies")
    parser.add_argument('--mlp_pth', type=str, 
                        default='/home/U116med/wch_code/dino_v3/models/polyp/vits16plus/concat_24681012_fine/best.pth',
                        help='MLP 模型權重路徑')
    parser.add_argument('--proto_pth', type=str,
                        default='/home/U116med/wch_code/dino_v3/models/prototype_cascade/vits16plus/baseline_prototype_fb_44_concat_fine/best.pth', 
                        help='Prototype 模型權重路徑')
    parser.add_argument('--data_path', type=str, 
                        default='/home/U116med/data/polyp/TestDataset/test/', 
                        help='測試集路徑')
    parser.add_argument('--output_dir', type=str, 
                        default='./model_comparison_top20/', 
                        help='輸出資料夾')
    parser.add_argument('--backbone', type=str, default='vits16plus', 
                        help='Backbone')
    parser.add_argument('--num_fg', type=int, default=4, 
                        help='前景原型數量')
    parser.add_argument('--num_bg', type=int, default=4, 
                        help='背景原型數量')
    parser.add_argument('--testsize', type=int, default=352)
    parser.add_argument('--num_cases', type=int, default=35)
    opt = parser.parse_args()

    os.makedirs(opt.output_dir, exist_ok=True)

    print("載入 MLP 模型...")
    # mlp_model = DINOv3_MLP(n_class=1, backbone_type=opt.backbone).cuda()
    mlp_model = DINOv3_MLP_concat(n_class=1, backbone_type=opt.backbone).cuda()
    mlp_model = load_weights_robustly(mlp_model, opt.mlp_pth)
    mlp_model.eval()

    print("載入 Prototype 模型...")
    # proto_model = DINOv3_Prototype(num_fg=opt.num_fg, num_bg=opt.num_bg, backbone_type=opt.backbone).cuda()
    proto_model = DINOv3_Prototype_concat(num_fg=opt.num_fg, num_bg=opt.num_bg, backbone_type=opt.backbone).cuda()
    proto_model = load_weights_robustly(proto_model, opt.proto_pth)
    proto_model.eval()

    test_loader = test_dataset(os.path.join(opt.data_path, 'images/'), 
                               os.path.join(opt.data_path, 'masks/'), 
                               opt.testsize)
    num_images = test_loader.size
    results_list = []

    print("執行雙模型推論與評估...")
    with torch.no_grad():
        for i in tqdm(range(num_images)):
            image_tensor, gt_mask, name = test_loader.load_data()
            
            gt_mask = np.asarray(gt_mask, np.float32)
            gt_mask /= (gt_mask.max() + 1e-8)
            image_tensor = image_tensor.cuda()
            gt_binary = np.where(gt_mask >= 0.5, 1, 0)
            target_flat = gt_binary.reshape(-1)
            smooth = 1.0

            # --------------------------------
            # MLP 推論
            # --------------------------------
            res_mlp = mlp_model(image_tensor)
            res_mlp = F.interpolate(res_mlp, size=gt_mask.shape, mode='bilinear', align_corners=False)
            res_mlp = res_mlp.sigmoid().data.cpu().numpy().squeeze()
            res_mlp = (res_mlp - res_mlp.min()) / (res_mlp.max() - res_mlp.min() + 1e-8)
            mlp_pred = np.where(res_mlp >= 0.5, 1, 0)
            
            intersection_mlp = (mlp_pred.reshape(-1) * target_flat).sum()
            mlp_dice = (2 * intersection_mlp + smooth) / (mlp_pred.sum() + gt_binary.sum() + smooth)

            # --------------------------------
            # Prototype 推論
            # --------------------------------
            logits_proto, sim_map_proto = proto_model(image_tensor)
            res_proto = (logits_proto[:, 1:2, :, :] - logits_proto[:, 0:1, :, :]) * 10.0
            res_proto = F.interpolate(res_proto, size=gt_mask.shape, mode='bilinear', align_corners=False)
            res_proto = res_proto.sigmoid().data.cpu().numpy().squeeze()
            res_proto = (res_proto - res_proto.min()) / (res_proto.max() - res_proto.min() + 1e-8)
            proto_pred = np.where(res_proto >= 0.5, 1, 0)
            
            sim_map_resized = F.interpolate(sim_map_proto, size=gt_mask.shape, mode='bilinear', align_corners=False)
            proto_k_map = sim_map_resized.argmax(dim=1).squeeze(0).cpu().numpy()

            intersection_proto = (proto_pred.reshape(-1) * target_flat).sum()
            proto_dice = (2 * intersection_proto + smooth) / (proto_pred.sum() + gt_binary.sum() + smooth)

            # --------------------------------
            # 記錄與計算差值
            # --------------------------------
            diff = abs(mlp_dice - proto_dice)
            
            results_list.append({
                'name': name,
                'mlp_dice': float(mlp_dice),
                'proto_dice': float(proto_dice),
                'diff': float(diff),
                'gt_mask': gt_binary,
                'mlp_pred': mlp_pred,
                'proto_pred': proto_pred,
                'proto_k_map': proto_k_map,
                'orig_img': tensor_to_bgr_image(image_tensor)
            })

    # 依照絕對差值由大到小排序 (降冪)
    results_list.sort(key=lambda x: x['diff'], reverse=True)
    top_divergent_cases = results_list[:opt.num_cases]
    
    print(f"\n生成差異最大的前 {opt.num_cases} 筆視覺化結果...")
    
    for rank, case in enumerate(top_divergent_cases, start=1):
        # 檔名標註誰勝出以及具體差值
        winner = "MLP" if case['mlp_dice'] > case['proto_dice'] else "Proto"
        save_name = f"Rank{rank:02d}_Diff{case['diff']:.4f}_{winner}Win_{case['name']}"
        save_path = os.path.join(opt.output_dir, save_name)
        
        save_divergent_comparison(
            img_bgr=case['orig_img'],
            gt_mask=case['gt_mask'],
            mlp_pred=case['mlp_pred'],
            proto_pred=case['proto_pred'],
            proto_k_map=case['proto_k_map'],
            mlp_dice=case['mlp_dice'],
            proto_dice=case['proto_dice'],
            diff=case['diff'],
            save_path=save_path,
            fg_num=opt.num_fg,
            bg_num=opt.num_bg
        )

    print(f"\n執行完畢，結果已儲存至：{opt.output_dir}")

if __name__ == '__main__':
    main()