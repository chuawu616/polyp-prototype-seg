import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm

from lib.networks_prototype_adaptive import DINOv3_Prototype_Adaptive
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

def generate_dynamic_palette(num_classes):
    """生成 N 種盡可能不重複的 BGR 顏色 (基於 HSV 空間均勻採樣)"""
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        hue = int(180 * i / num_classes)
        # 透過交替飽和度與明度來增加相鄰顏色的對比度
        sat = 255 if i % 2 == 0 else 180
        val = 255 if i % 3 != 0 else 200
        color_hsv = np.array([[[hue, sat, val]]], dtype=np.uint8)
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        palette[i] = color_bgr
    return palette

def save_adaptive_visualization(img_bgr, gt_mask, pred_mask, k_map, gate_weights, dice_score, save_path, palette):
    """
    生成五聯圖：原圖 | GT | Binary Pred | K-Class Map | All Gates Panel (多欄)
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
    vis_original = add_title_and_border(img_bgr.copy(), "Original")

    # [2] GT
    vis_gt = img_bgr.copy()
    overlay = vis_gt.copy()
    overlay[gt_mask == 1] = [0, 255, 255]
    vis_gt = cv2.addWeighted(overlay, 0.5, vis_gt, 0.5, 0)
    vis_gt = add_title_and_border(vis_gt, "Ground Truth")

    # [3] Binary Pred
    vis_pred = img_bgr.copy()
    overlay = vis_pred.copy()
    overlay[pred_mask == 1] = [0, 255, 0]
    vis_pred = cv2.addWeighted(overlay, 0.5, vis_pred, 0.5, 0)
    vis_pred = add_title_and_border(vis_pred, f"Pred (DICE: {dice_score:.4f})", text_color=(0, 0, 255))

    # [4] K-Class Map
    k_colored_map = np.zeros((h, w, 3), dtype=np.uint8)    
    for proto_idx in range(len(palette)):
        k_colored_map[k_map == proto_idx] = palette[proto_idx]
    vis_k = add_title_and_border(k_colored_map, "K-Class Segment")

    # [5] Gate Info Panel (列出全部 64 個 Prototype 的狀態)
    # 為了塞下所有資訊，我們將面板寬度加寬到 420 像素，並採用 4 欄佈局
    panel_w = 420
    info_panel = np.ones((h, panel_w, 3), dtype=np.uint8) * 255 
    cv2.putText(info_panel, "All Gate Activations (Sorted):", (10, 25), font, 0.55, (0, 0, 0), 2)
    
    # 根據門控權重排序 (從高到低)
    sorted_indices = np.argsort(gate_weights)[::-1]
    
    # 網格佈局計算
    num_items = len(gate_weights)
    cols = 4  # 分成 4 欄
    rows = int(np.ceil(num_items / cols)) # 每欄的行數 (64/4 = 16)
    
    col_width = panel_w // cols
    y_start = 50
    row_height = (h - y_start) // rows # 動態計算每行可用高度，確保不會超出邊界
    
    for i, idx in enumerate(sorted_indices):
        weight = gate_weights[idx]
        
        # 決定現在要畫在哪一欄(c)、哪一行(r)
        c = i // rows
        r = i % rows
        
        x_offset = 10 + c * col_width
        y_offset = y_start + r * row_height
        
        text = f"P{idx:02d}:{weight:.2f}"
        color = tuple(int(c) for c in palette[idx])
        
        # 如果權重極低 (<0.01)，文字變成灰色，方便一眼看出哪些被完全靜音
        text_color = (0, 0, 0) if weight >= 0.01 else (180, 180, 180)
        
        # 畫色塊
        cv2.rectangle(info_panel, (x_offset, y_offset-10), (x_offset+12, y_offset+2), color, -1)
        cv2.rectangle(info_panel, (x_offset, y_offset-10), (x_offset+12, y_offset+2), (50,50,50), 1)
        
        # 畫文字 (稍微縮小字體以適應密集排版)
        cv2.putText(info_panel, text, (x_offset+16, y_offset), font, 0.4, text_color, 1, cv2.LINE_AA)

    vis_info = add_title_and_border(info_panel, "Gate Analysis")

    # 橫向拼接五張圖片
    combined = np.concatenate((vis_original, vis_gt, vis_pred, vis_k, vis_info), axis=1)
    cv2.imwrite(save_path, combined)

def main():
    parser = argparse.ArgumentParser(description="Visualize Adaptive Prototype Gate Activations")
    parser.add_argument('--pth_path', type=str, 
                        default='/home/U116med/wch_code/dino_v3/models/prototype_cascade/vits16plus/adaptive_prototype_64/best.pth', 
                        help='Adaptive 模型權重路徑')
    parser.add_argument('--data_path', type=str, 
                        default='/home/U116med/data/polyp/TestDataset/test/', 
                        help='測試集路徑 (如 .../ETIS-LaribPolypDB)')
    parser.add_argument('--output_dir', type=str, default='./adaptive_gate_analysis_worst/', help='輸出資料夾')
    parser.add_argument('--backbone', type=str, default='vits16plus', help='Backbone')
    parser.add_argument('--pool_size', type=int, default=64, help='Prototype 總數')
    parser.add_argument('--testsize', type=int, default=352)
    parser.add_argument('--num_cases', type=int, default=40, help='要輸出幾筆結果')
    opt = parser.parse_args()

    os.makedirs(opt.output_dir, exist_ok=True)

    print("正在載入 Adaptive Prototype 模型...")
    model = DINOv3_Prototype_Adaptive(pool_size=opt.pool_size, backbone_type=opt.backbone, layer_indices=(1,3,5,7,9,11)).cuda()
    
    checkpoint = torch.load(opt.pth_path, map_location='cuda')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    image_root = os.path.join(opt.data_path, 'images/')
    gt_root = os.path.join(opt.data_path, 'masks/')
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    num_images = test_loader.size
    
    # 建立調色盤
    palette = generate_dynamic_palette(opt.pool_size)
    results_list = []

    print("開始進行推論與特徵萃取...")
    with torch.no_grad():
        for i in tqdm(range(num_images)):
            image_tensor, gt_mask, name = test_loader.load_data()
            
            gt_mask = np.asarray(gt_mask, np.float32)
            gt_mask /= (gt_mask.max() + 1e-8)
            image_tensor = image_tensor.cuda()
            gt_binary = np.where(gt_mask >= 0.5, 1, 0)
            
            # 模型推論
            logits_high, gated_sim_map_high, gate_weights = model(image_tensor)
            
            # 二值化預測
            res = F.interpolate(logits_high, size=gt_mask.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            pred_binary = np.where(res >= 0.5, 1, 0)
            
            # 取得 K-Class Map (只考慮相似度最高的那一個 Prototype)
            sim_map_resized = F.interpolate(gated_sim_map_high, size=gt_mask.shape, mode='bilinear', align_corners=False)
            k_map = sim_map_resized.argmax(dim=1).squeeze(0).cpu().numpy()
            
            # 取得該圖片的 Gate 權重陣列
            gate_weights_np = gate_weights.squeeze(0).cpu().numpy()

            # 計算 DICE
            smooth = 1.0
            intersection = (pred_binary.reshape(-1) * gt_binary.reshape(-1)).sum()
            dice = (2 * intersection + smooth) / (pred_binary.sum() + gt_binary.sum() + smooth)

            results_list.append({
                'name': name,
                'dice': float(dice),
                'gt_mask': gt_binary,
                'pred_mask': pred_binary,
                'k_map': k_map,
                'gate_weights': gate_weights_np,
                'orig_img': tensor_to_bgr_image(image_tensor)
            })

    # 依照 DICE 升冪排序 (最差的排前面)
    results_list.sort(key=lambda x: x['dice'])
    worst_cases = results_list[:opt.num_cases]
    
    print(f"\n生成 DICE 表現最差的 {opt.num_cases} 筆視覺化結果...")
    
    for rank, case in enumerate(worst_cases, start=1):
        save_name = f"Rank{rank:02d}_DICE_{case['dice']:.4f}_{case['name']}"
        save_path = os.path.join(opt.output_dir, save_name)
        
        save_adaptive_visualization(
            img_bgr=case['orig_img'],
            gt_mask=case['gt_mask'],
            pred_mask=case['pred_mask'],
            k_map=case['k_map'],
            gate_weights=case['gate_weights'],
            dice_score=case['dice'],
            save_path=save_path,
            palette=palette
        )

    print(f"\n執行完畢，請至 {opt.output_dir} 查看結果。")

if __name__ == '__main__':
    main()