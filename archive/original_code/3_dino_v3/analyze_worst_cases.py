import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
from lib.networks import DINOv3_MLP
from lib.networks_concat import DINOv3_MLP_concat
from utils.dataloader import test_dataset

def tensor_to_bgr_image(tensor):
    """將 Normalize 過的 Tensor 還原回可視化的 BGR 圖片"""
    tensor = tensor.squeeze(0).cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    img = tensor * std + mean
    img = torch.clamp(img, 0, 1)
    img_np = img.permute(1, 2, 0).numpy()
    
    img_bgr = (img_np * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
    return img_bgr

def save_comparison_image(img_bgr, gt_mask, pred_mask, dice_score, save_path):
    """將原圖、GT與預測結果拼成一張圖，並標註 DICE 分數"""
    h, w = gt_mask.shape[:2]
    
    # 關鍵修正：確保原圖被放大回與 GT 相同的原始尺寸
    if img_bgr.shape[:2] != (h, w):
        img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        
    # 將單通道的 0/1 Mask 轉換為 3 通道的 0/255 影像以便拼接
    gt_color = cv2.cvtColor((gt_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    pred_color = cv2.cvtColor((pred_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    # 加上標題字樣
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_bgr, "Original", (10, 30), font, 0.8, (0, 255, 0), 2)
    cv2.putText(gt_color, "Ground Truth", (10, 30), font, 0.8, (0, 255, 0), 2)
    cv2.putText(pred_color, f"Pred (DICE: {dice_score:.4f})", (10, 30), font, 0.8, (0, 0, 255), 2)
    
    # 橫向拼接三張圖片
    combined = np.concatenate((img_bgr, gt_color, pred_color), axis=1)
    cv2.imwrite(save_path, combined)

def main():
    parser = argparse.ArgumentParser(description="Find and Visualize the worst 20 cases based on DICE score")
    parser.add_argument('--pth_path', type=str, 
                        default='/home/U116med/wch_code/dino_v3/models/polyp/vits16plus/concat_24681012/best.pth', 
                        help='訓練好的 best.pth 路徑')
    parser.add_argument('--data_path', type=str, 
                        default='/home/U116med/data/polyp/TestDataset/test',                        
                        help='測試資料集的路徑 (例如 .../CVC-ClinicDB)')
    parser.add_argument('--output_dir', type=str, 
                        default='./mlp_worst_40_cases_baseline_fine/', 
                        help='最差結果的儲存資料夾')
    parser.add_argument('--backbone', type=str, 
                        default='vits16plus', 
                        help='使用的 DINOv3 版本')
    parser.add_argument('--testsize', type=int, 
                        default=352, 
                        help='推論尺寸')
    parser.add_argument('--num_cases', type=int, 
                        default=40, 
                        help='要挑選出幾筆最差的資料')
    opt = parser.parse_args()

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    print("正在載入模型...")
    # model = DINOv3_MLP(n_class=1, backbone_type=opt.backbone).cuda()
    model = DINOv3_MLP_concat(n_class=1, backbone_type=opt.backbone).cuda()

    model.load_state_dict(torch.load(opt.pth_path, map_location='cuda'), strict=True)
    model.eval()

    image_root = os.path.join(opt.data_path, 'images/')
    gt_root = os.path.join(opt.data_path, 'masks/')
    
    print(f"載入測試資料集: {opt.data_path}")
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    num_images = test_loader.size
    
    results_list = []

    print("開始進行全資料集推論與 DICE 計算...")
    with torch.no_grad():
        for i in tqdm(range(num_images)):
            # 讀取資料
            image_tensor, gt_mask, name = test_loader.load_data()
            
            gt_mask = np.asarray(gt_mask, np.float32)
            gt_mask /= (gt_mask.max() + 1e-8)
            image_tensor = image_tensor.cuda()

            # 模型推論
            res = model(image_tensor)
            res = F.interpolate(res, size=gt_mask.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            
            # 產生二值化 Mask
            pred_binary = np.where(res >= 0.5, 1, 0)
            gt_binary = np.where(gt_mask >= 0.5, 1, 0)
            
            # 計算 DICE 分數
            smooth = 1
            intersection = (pred_binary.reshape(-1) * gt_binary.reshape(-1)).sum()
            dice = (2 * intersection + smooth) / (pred_binary.sum() + gt_binary.sum() + smooth)
            
            # 為了避免記憶體爆炸，我們只在記憶體中保留輕量的二值化矩陣與分數
            results_list.append({
                'name': name,
                'dice': float(dice),
                'pred_mask': pred_binary,
                'gt_mask': gt_binary,
                'orig_img': tensor_to_bgr_image(image_tensor) # 保存用於視覺化的原圖
            })

    # 根據 DICE 分數進行升冪排序 (分數越低排越前面)
    results_list.sort(key=lambda x: x['dice'])
    
    # 提取前 20 名最差的案例
    worst_cases = results_list[:opt.num_cases]
    
    print(f"\n已找出 DICE 表現最低的 {opt.num_cases} 筆資料，正在生成比對圖...")
    
    for rank, case in enumerate(worst_cases, start=1):
        print(f"Rank {rank:02d} | DICE: {case['dice']:.4f} | File: {case['name']}")
        
        # 組合儲存檔名，加入排名與分數，方便在資料夾中直接排序觀察
        save_name = f"Rank{rank:02d}_DICE_{case['dice']:.4f}_{case['name']}"
        save_path = os.path.join(opt.output_dir, save_name)
        
        save_comparison_image(
            img_bgr=case['orig_img'],
            gt_mask=case['gt_mask'],
            pred_mask=case['pred_mask'],
            dice_score=case['dice'],
            save_path=save_path
        )

    print(f"\n任務完成！請至 {opt.output_dir} 資料夾查看結果。")

if __name__ == '__main__':
    main()