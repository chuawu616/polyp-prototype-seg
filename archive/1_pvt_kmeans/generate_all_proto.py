import os
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import torch.nn.functional as F
import random
import models.pvtv2 as pvt
from dataloaders.SupervisedPolypDataset import SupervisedPolypDataset

def analyze_per_image_prototypes(cfg, all_prototypes, all_paths):
    """
    對每一張圖像的 16 個局部原型進行相似度/距離分析，
    匯總全局統計結果，並隨機展示 N 筆詳細的單圖分析。
    """
    print("\n--- 開始逐圖分析原型相似度與距離 ---")
    
    all_fg_vs_fg_sim, all_bg_vs_bg_sim, all_fg_vs_bg_sim = [], [], []
    all_fg_vs_fg_dist, all_bg_vs_bg_dist, all_fg_vs_bg_dist = [], [], []
    num_fg, num_bg = cfg['n_fg_prototypes'], cfg['n_bg_prototypes']
    
    # --- 新增點：用於儲存每張圖的獨立統計結果 ---
    per_image_stats = []

    if len(all_prototypes) != len(all_paths):
        print("警告: 原型數量與路徑數量不匹配，無法進行逐圖分析。")
        return

    for i, per_image_prototypes in enumerate(tqdm(all_prototypes, desc="分析中")):
        if per_image_prototypes is None: 
            per_image_stats.append(None) # 添加佔位符
            continue
            
        cos_sim_matrix = cosine_similarity(per_image_prototypes)
        euclidean_dist_matrix = euclidean_distances(per_image_prototypes)

        current_stats = {}
        
        if num_fg > 1:
            fg_vs_fg_sim = cos_sim_matrix[:num_fg, :num_fg][np.triu_indices(num_fg, k=1)]
            fg_vs_fg_dist = euclidean_dist_matrix[:num_fg, :num_fg][np.triu_indices(num_fg, k=1)]
            current_stats['fg_sim_mean'] = np.mean(fg_vs_fg_sim)
            current_stats['fg_dist_mean'] = np.mean(fg_vs_fg_dist)
            all_fg_vs_fg_sim.extend(fg_vs_fg_sim)
            all_fg_vs_fg_dist.extend(fg_vs_fg_dist)

        if num_bg > 1:
            bg_vs_bg_sim = cos_sim_matrix[num_fg:, num_fg:][np.triu_indices(num_bg, k=1)]
            bg_vs_bg_dist = euclidean_dist_matrix[num_fg:, num_fg:][np.triu_indices(num_bg, k=1)]
            current_stats['bg_sim_mean'] = np.mean(bg_vs_bg_sim)
            current_stats['bg_dist_mean'] = np.mean(bg_vs_bg_dist)
            all_bg_vs_bg_sim.extend(bg_vs_bg_sim)
            all_bg_vs_bg_dist.extend(bg_vs_bg_dist)
            
        fg_vs_bg_sim = cos_sim_matrix[:num_fg, num_fg:].flatten()
        fg_vs_bg_dist = euclidean_dist_matrix[:num_fg, num_fg:].flatten()
        current_stats['fg_bg_sim_mean'] = np.mean(fg_vs_bg_sim)
        current_stats['fg_bg_dist_mean'] = np.mean(fg_vs_bg_dist)
        all_fg_vs_bg_sim.extend(fg_vs_bg_sim)
        all_fg_vs_bg_dist.extend(fg_vs_bg_dist)
        
        per_image_stats.append(current_stats)

    mean_fg_fg_sim, std_fg_fg_sim = np.mean(all_fg_vs_fg_sim), np.std(all_fg_vs_fg_sim)
    mean_bg_bg_sim, std_bg_bg_sim = np.mean(all_bg_vs_bg_sim), np.std(all_bg_vs_bg_sim)
    mean_fg_bg_sim, std_fg_bg_sim = np.mean(all_fg_vs_bg_sim), np.std(all_fg_vs_bg_sim)
    mean_fg_fg_dist, std_fg_fg_dist = np.mean(all_fg_vs_fg_dist), np.std(all_fg_vs_fg_dist)
    mean_bg_bg_dist, std_bg_bg_dist = np.mean(all_bg_vs_bg_dist), np.std(all_bg_vs_bg_dist)
    mean_fg_bg_dist, std_fg_bg_dist = np.mean(all_fg_vs_bg_dist), np.std(all_fg_vs_bg_dist)

    # --- 將結果寫入文字檔案 ---
    output_analysis_path = cfg['output_analysis_file']
    with open(output_analysis_path, 'w') as f:
        f.write("逐圖原型相似度分析報告\n")
        f.write("==================================\n\n")
        
        # --- 寫入全局統計摘要 ---
        f.write("--- 全局統計摘要 ---\n")
        f.write("註: 以下所有指標都是基於數據集上所有圖像的逐圖計算結果的統計。\n\n")
        f.write("--- 餘弦相似度 (Cosine Similarity) --- (值越高越相似)\n")
        f.write(f"前景原型內部平均相似度: {mean_fg_fg_sim:.4f} (+/- {std_fg_fg_sim:.4f})\n")
        f.write(f"背景原型內部平均相似度: {mean_bg_bg_sim:.4f} (+/- {std_bg_bg_sim:.4f})\n")
        f.write(f"前景 vs 背景原型平均相似度: {mean_fg_bg_sim:.4f} (+/- {std_fg_bg_sim:.4f})  <--- 期望此值較低\n\n")
        f.write("--- 歐幾里得距離 (Euclidean Distance) --- (值越低越相似)\n")
        f.write(f"前景原型內部平均距離: {mean_fg_fg_dist:.4f} (+/- {std_fg_fg_dist:.4f})\n")
        f.write(f"背景原型內部平均距離: {mean_bg_bg_dist:.4f} (+/- {std_bg_bg_dist:.4f})\n")
        f.write(f"前景 vs 背景原型平均距離: {mean_fg_bg_dist:.4f} (+/- {std_fg_bg_dist:.4f})  <--- 期望此值較高\n\n")
        
        # --- 新增點：隨機選取 N 筆單圖詳細結果並寫入 ---
        f.write("==================================\n")
        f.write(f"--- 隨機 {cfg['num_detailed_samples']} 筆單圖分析詳情 ---\n\n")
        
        num_samples = len(all_paths)
        sample_indices = random.sample(range(num_samples), k=min(num_samples, cfg['num_detailed_samples']))
        
        for idx in sample_indices:
            path = all_paths[idx]
            stats = per_image_stats[idx]
            base_name = os.path.basename(path)
            
            f.write(f"--- 圖像: {base_name} ---\n")
            if stats:
                f.write(f"  - 前景內部相似度/距離: {stats.get('fg_sim_mean', 'N/A'):.4f} / {stats.get('fg_dist_mean', 'N/A'):.4f}\n")
                f.write(f"  - 背景內部相似度/距離: {stats.get('bg_sim_mean', 'N/A'):.4f} / {stats.get('bg_dist_mean', 'N/A'):.4f}\n")
                f.write(f"  - 前景 vs 背景相似度/距離: {stats.get('fg_bg_sim_mean', 'N/A'):.4f} / {stats.get('fg_bg_dist_mean', 'N/A'):.4f}\n\n")
            else:
                f.write("  - (無有效原型可供分析)\n\n")
    
    print(f"分析報告已成功儲存至: {output_analysis_path}")

def generate_per_image_prototypes(cfg):
    device = torch.device(f"cuda:{cfg['gpu_id']}" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    
    model = pvt.pvt_v2_b2(pretrained=True).to(device).eval()
    dataset = SupervisedPolypDataset(
        image_dir=cfg['image_dir'],
        mask_dir=cfg['mask_dir'],
        target_size=(cfg['image_size'], cfg['image_size']),
        return_original=True # 假設 Dataset 返回 4 個值
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)

    output_dir = cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    print(f"局部原型將被儲存至: {output_dir}")

    all_prototypes_for_analysis = []
    all_paths_for_analysis = []

    with torch.no_grad():
        for image, mask, prototype_tensor, original_image, path in tqdm(dataloader, desc="生成局部原型"):
            base_name = os.path.splitext(os.path.basename(path[0]))[0]
            output_path = os.path.join(output_dir, f"{base_name}_protos.npy")
            all_paths_for_analysis.append(path[0])

            if os.path.exists(output_path) and not cfg.get('overwrite', False):
                prototypes = np.load(output_path)
                all_prototypes_for_analysis.append(prototypes)
                continue

            image = image.to(device)
            features = model.forward_features(image)[cfg['target_stage_idx']]
            masks_small = F.interpolate(mask.unsqueeze(1).float(), size=features.shape[-2:], mode='nearest').squeeze(1)
            
            feat_map = features[0].permute(1, 2, 0)
            true_mask = masks_small[0]
            
            pixels_flat = feat_map.view(-1, feat_map.shape[-1])
            mask_flat = true_mask.view(-1)

            fg_pixels = pixels_flat[mask_flat == 1].cpu().numpy()
            bg_pixels = pixels_flat[mask_flat == 0].cpu().numpy()
            
            fg_centroids, bg_centroids = None, None
            
            if len(fg_pixels) >= cfg['n_fg_prototypes']:
                kmeans_fg = KMeans(n_clusters=cfg['n_fg_prototypes'], random_state=42, n_init=10)
                kmeans_fg.fit(fg_pixels)
                fg_centroids = kmeans_fg.cluster_centers_
            
            if len(bg_pixels) >= cfg['n_bg_prototypes']:
                kmeans_bg = KMeans(n_clusters=cfg['n_bg_prototypes'], random_state=42, n_init=10)
                kmeans_bg.fit(bg_pixels)
                bg_centroids = kmeans_bg.cluster_centers_

            feature_dim = feat_map.shape[-1]
            if fg_centroids is None: fg_centroids = np.zeros((cfg['n_fg_prototypes'], feature_dim))
            if bg_centroids is None: bg_centroids = np.zeros((cfg['n_bg_prototypes'], feature_dim))
            
            per_image_prototypes = np.concatenate([fg_centroids, bg_centroids], axis=0)
            np.save(output_path, per_image_prototypes)
            all_prototypes_for_analysis.append(per_image_prototypes)

    print("\n所有圖像的局部原型已生成並儲存。")
    
    if cfg.get('run_analysis', True):
        analyze_per_image_prototypes(cfg, all_prototypes_for_analysis, all_paths_for_analysis)

if __name__ == '__main__':
    config = {
        'gpu_id': 0,
        'image_dir': '/home/U116med/data/polyp/TestDataset/test/images/',
        'mask_dir': '/home/U116med/data/polyp/TestDataset/test/masks/',
        'image_size': 352,
        'batch_size': 1,
        'overwrite': False,
        
        'target_stage_idx': 1,
        
        'n_fg_prototypes': 8,
        'n_bg_prototypes': 8,

        'output_dir': './valid_protos_8_8/',
        'output_analysis_file': 'valid_prototypes_analysis_detailed.txt',
        'run_analysis': True,
        
        'num_detailed_samples': 20
    }
    
    generate_per_image_prototypes(config)