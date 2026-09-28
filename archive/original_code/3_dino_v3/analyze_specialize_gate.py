import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from lib.networks_prototype import DINOv3_Prototype
from lib.networks_prototype_concat import DINOv3_Prototype_concat
from lib.networks_prototype_specialize import DINOv3_Prototype_Specialize

from lib.networks_prototype_specialize_pvtv2 import Prototype_CASCADE_Specialize
from utils.dataloader import test_dataset

def main():
    parser = argparse.ArgumentParser(description="Analyze Gating Weight (prob_easy) Histogram")
    parser.add_argument('--pth_path', type=str, 
                        default='/home/U116med/wch_code/cascade/models/prototype_cascade/baseline_prototype_fhb_524_warm_10/best.pth', 
                        help='訓練好的 Specialize 模型 weights 路徑')
    parser.add_argument('--data_path', type=str, default='/home/U116med/data/polyp/TestDataset/test', help='測試資料集路徑')
    parser.add_argument('--output_dir', type=str, default='./analysis_results/', help='圖表儲存目錄')
    parser.add_argument('--backbone', type=str, default='vits16plus', help='DINOv3 版本')
    parser.add_argument('--testsize', type=int, default=352, help='推論尺寸')
    parser.add_argument('--num_fg', type=int, default=5, help='前景原型總數量')
    parser.add_argument('--num_hard', type=int, default=2, help='困難原型數量')
    parser.add_argument('--num_bg', type=int, default=4, help='背景原型數量')
    opt = parser.parse_args()

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    print(f"正在載入模型: {opt.pth_path}")
    # model = DINOv3_Prototype_Specialize(num_fg=opt.num_fg, num_hard=opt.num_hard, num_bg=opt.num_bg, backbone_type=opt.backbone).cuda()
    model = Prototype_CASCADE_Specialize(num_fg=opt.num_fg, num_hard=opt.num_hard, num_bg=opt.num_bg).cuda()
    
    checkpoint = torch.load(opt.pth_path, map_location='cuda')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    image_root = os.path.join(opt.data_path, 'images/')
    gt_root = os.path.join(opt.data_path, 'masks/')
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    
    # 初始化 Histogram Bins (0 到 1 切成 100 等份)
    num_bins = 100
    bins = np.linspace(0, 1, num_bins + 1)
    
    # 用來累積所有測試圖片的像素數量
    hist_fg_counts = np.zeros(num_bins, dtype=np.float64)
    hist_bg_counts = np.zeros(num_bins, dtype=np.float64)
    hist_all_counts = np.zeros(num_bins, dtype=np.float64)

    print(f"開始提取特徵與累積直方圖 ({test_loader.size} 張圖片)...")
    
    with torch.no_grad():
        for i in tqdm(range(test_loader.size)):
            image_tensor, gt_mask, _ = test_loader.load_data()
            
            # GT 二值化
            gt_np = np.asarray(gt_mask, np.float32)
            gt_np /= (gt_np.max() + 1e-8)
            gt_binary = np.where(gt_np >= 0.5, 1, 0).astype(bool)
            
            image_tensor = image_tensor.cuda()
            
            # 模型推論取得相似度特徵圖 (B, K, H, W)
            _, similarity_map_high = model(image_tensor)
            
            # --- 重建 prob_easy 矩陣 ---
            num_easy = opt.num_fg - 1
            sim_easy = similarity_map_high[:, :num_easy, :, :]
            sim_bg = similarity_map_high[:, opt.num_fg:, :, :]
            
            score_easy = torch.logsumexp(sim_easy, dim=1)
            score_bg = torch.logsumexp(sim_bg, dim=1)
            
            # 計算門控權重 (H, W)
            prob_easy = torch.sigmoid(score_easy - score_bg).squeeze().cpu().numpy()
            
            # 確保尺寸一致
            if prob_easy.shape != gt_binary.shape:
                import cv2
                prob_easy = cv2.resize(prob_easy, (gt_binary.shape[1], gt_binary.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            # --- 分割為前景與背景像素陣列 ---
            prob_easy_fg = prob_easy[gt_binary]       # 落在真實息肉內的像素
            prob_easy_bg = prob_easy[~gt_binary]      # 落在真實背景內的像素
            
            # 累積 Histogram
            counts_fg, _ = np.histogram(prob_easy_fg, bins=bins)
            counts_bg, _ = np.histogram(prob_easy_bg, bins=bins)
            counts_all, _ = np.histogram(prob_easy.flatten(), bins=bins)
            
            hist_fg_counts += counts_fg
            hist_bg_counts += counts_bg
            hist_all_counts += counts_all

    # --- 繪製與儲存圖表 ---
    print("\n繪製分佈圖中...")
    
    # 為了避免背景像素量遠大於前景導致 Y 軸比例失衡，轉換為機率密度 (Density/Percentage)
    hist_fg_density = hist_fg_counts / (hist_fg_counts.sum() + 1e-8)
    hist_bg_density = hist_bg_counts / (hist_bg_counts.sum() + 1e-8)
    hist_all_density = hist_all_counts / (hist_all_counts.sum() + 1e-8)
    
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 圖 1: 全局像素分佈
    axes[0].bar(bin_centers, hist_all_density, width=0.01, color='gray', alpha=0.7)
    axes[0].set_title("Overall prob_easy Distribution")
    axes[0].set_xlabel("prob_easy (Gate Value)")
    axes[0].set_ylabel("Density")
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 圖 2: 僅限真實息肉區域 (Foreground)
    axes[1].bar(bin_centers, hist_fg_density, width=0.01, color='red', alpha=0.7)
    axes[1].set_title("Foreground (Polyp) Pixels Only")
    axes[1].set_xlabel("prob_easy (Gate Value)")
    axes[1].set_xlim(-0.05, 1.05)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 圖 3: 僅限真實背景區域 (Background)
    axes[2].bar(bin_centers, hist_bg_density, width=0.01, color='blue', alpha=0.7)
    axes[2].set_title("Background Pixels Only")
    axes[2].set_xlabel("prob_easy (Gate Value)")
    axes[2].set_xlim(-0.05, 1.05)
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_name = os.path.basename(opt.pth_path).replace('.pth', '_gate_histogram.png')
    save_path = os.path.join(opt.output_dir, save_name)
    plt.savefig(save_path, dpi=300)
    
    print(f"直方圖已儲存至: {save_path}")

    # --- 自動分析報告 ---
    print("\n--- 門控權重分佈診斷 ---")
    
    def analyze_distribution(density_array, name):
        # 統計極端區與模糊區的比例
        hard_zone = density_array[:10].sum()   # 0.0 ~ 0.1 (完全依賴 Hard)
        easy_zone = density_array[-10:].sum()  # 0.9 ~ 1.0 (完全依賴 Easy)
        fuzzy_zone = density_array[30:70].sum() # 0.3 ~ 0.7 (模糊過渡區)
        
        print(f"[{name} 區域分佈]")
        print(f"  - 依賴 Hard (prob < 0.1): {hard_zone*100:.2f}%")
        print(f"  - 模糊過渡 (0.3 < prob < 0.7): {fuzzy_zone*100:.2f}%")
        print(f"  - 依賴 Easy (prob > 0.9): {easy_zone*100:.2f}%")
        
        if fuzzy_zone < 0.01 and (hard_zone > 0.9 or easy_zone > 0.9):
            print(f"警告: {name} 區域發生門控飽和 (Hard/Easy 開關過於極端)。")
        return hard_zone, easy_zone

    analyze_distribution(hist_fg_density, "前景 (息肉)")
    print("-" * 30)
    analyze_distribution(hist_bg_density, "背景 (腸道)")

if __name__ == '__main__':
    main()