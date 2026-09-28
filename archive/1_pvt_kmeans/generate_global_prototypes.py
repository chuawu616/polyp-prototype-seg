import os
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import faiss
from sklearn.cluster import KMeans
from skimage.segmentation import mark_boundaries
from skimage import color
import torch.nn.functional as F
import cv2

import models.pvtv2 as pvt
from dataloaders.SupervisedPolypDataset import SupervisedPolypDataset

def custom_collate_for_protogen(batch):
    """
    一個自訂的 collate_fn，專門用於原型生成。
    它能處理尺寸不一的 'original_image'。
    """
    # 對於尺寸一致的 'images' 和 'masks'，使用 PyTorch 預設的 collate 邏輯
    images = torch.stack([item[0] for item in batch], 0)
    masks = torch.stack([item[1] for item in batch], 0)
    
    # 對於尺寸不一的 'original_images' 和 'paths'，將它們收集到 list 中
    original_images = [item[2] for item in batch]
    paths = [item[3] for item in batch]
    
    return images, masks, original_images, paths

def save_visualization(output_dir, base_name, original_image, kmeans_fg_map, kmeans_bg_map):
    """
    儲存用於檢查第一階段 K-Means 結果的視覺化圖像。
    """
    # 確保 original_image 是 (H, W, C) 的 uint8 格式
    if original_image.dtype != np.uint8:
        original_image = (original_image * 255).clip(0, 255).astype(np.uint8)
    if original_image.shape[0] < original_image.shape[2]: # 如果是 (C, H, W)
        original_image = np.transpose(original_image, (1, 2, 0))

    # --- 修正點：在上色前，將低解析度的 K-Means 圖上採樣回原圖尺寸 ---
    h, w, _ = original_image.shape
    if kmeans_fg_map.shape != (h, w):
        kmeans_fg_map = cv2.resize(kmeans_fg_map.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    if kmeans_bg_map.shape != (h, w):
        kmeans_bg_map = cv2.resize(kmeans_bg_map.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    kmeans_fg_vis = color.label2rgb(kmeans_fg_map, image=original_image, bg_label=0, image_alpha=0.5)
    kmeans_bg_vis = color.label2rgb(kmeans_bg_map, image=original_image, bg_label=0, image_alpha=0.5)
    kmeans_fg_vis = (kmeans_fg_vis * 255).astype(np.uint8)
    kmeans_bg_vis = (kmeans_bg_vis * 255).astype(np.uint8)
    combined_kmeans_map = kmeans_fg_map.copy()
    bg_mask = kmeans_bg_map > 0
    combined_kmeans_map[bg_mask] = kmeans_bg_map[bg_mask] + np.max(kmeans_fg_map)
    combined_kmeans_vis = color.label2rgb(combined_kmeans_map, image=original_image, bg_label=0, image_alpha=0.5)
    combined_kmeans_vis = (combined_kmeans_vis * 255).astype(np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    def add_title(image, text):
        img_with_title = cv2.copyMakeBorder(image, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.putText(img_with_title, text, (10, 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        return img_with_title
    img1 = add_title(cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR), "Original")
    img2 = add_title(cv2.cvtColor(combined_kmeans_vis, cv2.COLOR_RGB2BGR), "Per-Image KMeans")
    img3 = add_title(cv2.cvtColor(kmeans_fg_vis, cv2.COLOR_RGB2BGR), "FG KMeans")
    img4 = add_title(cv2.cvtColor(kmeans_bg_vis, cv2.COLOR_RGB2BGR), "BG KMeans")
    combined_image = np.concatenate((img1, img2, img3, img4), axis=1)
    output_path = os.path.join(output_dir, f"{base_name}_check.png")
    cv2.imwrite(output_path, combined_image)

def run_faiss_kmeans(data, n_clusters, gpu_id=0):
    """
    使用 Faiss 在 GPU 上運行 K-Means。
    
    Args:
        data (np.array): 形狀為 (N, D) 的數據。
        n_clusters (int): 聚類數量 K。
        gpu_id (int): 使用的 GPU ID。
    
    Returns:
        np.array: 形狀為 (K, D) 的中心點。
    """
    print(f"正在 {data.shape[0]} 個點上運行 Faiss K-Means，聚為 {n_clusters} 類...")
    n_samples, dim = data.shape
    
    # 創建一個 K-Means 物件
    # nredo=5 表示演算法會用不同的隨機種子跑5次，返回最好的結果
    kmeans = faiss.Kmeans(d=dim, k=n_clusters, niter=20, verbose=True, gpu=gpu_id, nredo=5)
    
    # 訓練
    kmeans.train(data.astype(np.float32))
    
    # 返回中心點
    return kmeans.centroids

def generate_prototypes(cfg):
    device = torch.device(f"cuda:{cfg['gpu_id']}" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    
    # --- 1. 準備模型和資料 ---
    model = pvt.pvt_v2_b2(pretrained=True).to(device).eval()
    
    dataset = SupervisedPolypDataset(
        image_dir=cfg['image_dir'],
        mask_dir=cfg['mask_dir'],
        target_size=(cfg['image_size'], cfg['image_size']),
        return_original=True        
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=cfg['batch_size'], 
        num_workers=4,
        collate_fn=custom_collate_for_protogen
    )

    vis_dir = cfg.get('visualization_dir')
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    # --- 2. 階段一: 在低解析度特徵上計算每張圖的 K-Means 中心 ---
    print("階段一: 在低解析度 (44x44) 特徵上計算局部 K-Means 中心...")
    all_fg_image_centroids = []
    all_bg_image_centroids = []

    with torch.no_grad():
        for batch_idx, (images, masks, original_images, paths) in enumerate(tqdm(dataloader, desc="階段一: 處理批次")):
            images = images.to(device)
            
            # a. 提取 44x44x128 特徵
            features = model.forward_features(images)[cfg['target_stage_idx']] # (B, 128, 44, 44)
            
            # b. **核心修改點**: 將真實 Mask 下採樣到 44x44
            masks_small = F.interpolate(masks.unsqueeze(1).float(), 
                                        size=features.shape[-2:], 
                                        mode='nearest').squeeze(1) # (B, 44, 44)
            
            # --- 優化點：盡可能在 GPU 上操作，只在 K-Means 前移到 CPU ---
            features = features.permute(0, 2, 3, 1) # (B, 44, 44, C) on GPU

            for i in range(images.shape[0]):
                feat_map = features[i] # (44, 44, C), on GPU
                true_mask = masks_small[i] # (44, 44), on GPU
                
                pixels_flat = feat_map.view(-1, feat_map.shape[-1])
                mask_flat = true_mask.view(-1)

                fg_pixels = pixels_flat[mask_flat == 1]
                bg_pixels = pixels_flat[mask_flat == 0]
                
                # 將數據移到 CPU 準備 K-Means (scikit-learn 在 CPU 上運行)
                fg_pixels_np = fg_pixels.cpu().numpy()
                bg_pixels_np = bg_pixels.cpu().numpy()
                
                # 初始化 K-Means 結果圖 (在低解析度上)
                kmeans_fg_map = np.zeros(true_mask.shape, dtype=np.uint8)
                kmeans_bg_map = np.zeros(true_mask.shape, dtype=np.uint8)
                
                # 對前景像素做 K-Means
                if len(fg_pixels_np) >= cfg['n_fg_prototypes_per_image']:
                    # 使用 scikit-learn 進行圖像內的聚類，因為數據量不大
                    kmeans_fg = KMeans(n_clusters=cfg['n_fg_prototypes_per_image'], random_state=42, n_init=10)
                    fg_labels = kmeans_fg.fit_predict(fg_pixels_np)
                    all_fg_image_centroids.extend(kmeans_fg.cluster_centers_)
                    # 將 K-Means 標籤填回低解析度圖
                    kmeans_fg_map[true_mask.cpu().numpy() == 1] = fg_labels + 1
                
                # 對背景像素做 K-Means
                if len(bg_pixels_np) >= cfg['n_bg_prototypes_per_image']:
                    kmeans_bg = KMeans(n_clusters=cfg['n_bg_prototypes_per_image'], random_state=42, n_init=10)
                    bg_labels = kmeans_bg.fit_predict(bg_pixels_np)
                    all_bg_image_centroids.extend(kmeans_bg.cluster_centers_)
                    kmeans_bg_map[true_mask.cpu().numpy() == 0] = bg_labels + 1

                # 輸出前10張圖像的檢查圖
                current_image_index = batch_idx * cfg['batch_size'] + i
                if vis_dir and current_image_index < 10:
                    original_image_np = original_images[i]
                    base_name = os.path.splitext(os.path.basename(paths[i]))[0]
                    save_visualization(vis_dir, base_name, original_image_np, kmeans_fg_map, kmeans_bg_map)

    # --- 3. 階段二: 使用 Faiss-GPU 在全局中心池上進行最終聚類 ---
    print(f"\n階段一完成: 共收集到 {len(all_fg_image_centroids)} 個前景中心和 {len(all_bg_image_centroids)} 個背景中心。")
    if len(all_fg_image_centroids) < cfg['n_fg_prototypes'] or len(all_bg_image_centroids) < cfg['n_bg_prototypes']:
        print("錯誤: 收集到的中心點數量不足。")
        return
        
    print("階段二: 進行全局 GPU K-Means 聚類...")
    fg_centroids_pool = np.array(all_fg_image_centroids)
    bg_centroids_pool = np.array(all_bg_image_centroids)

    # --- 使用 Faiss 進行 GPU K-Means ---
    final_fg_prototypes = run_faiss_kmeans(fg_centroids_pool, cfg['n_fg_prototypes'], gpu_id=cfg['gpu_id'])
    final_bg_prototypes = run_faiss_kmeans(bg_centroids_pool, cfg['n_bg_prototypes'], gpu_id=cfg['gpu_id'])

    print(f"前景全局原型計算完成，形狀: {final_fg_prototypes.shape}")
    print(f"背景全局原型計算完成，形狀: {final_bg_prototypes.shape}")
    
    # --- 4. 組合併保存 ---
    all_prototypes = np.concatenate([final_fg_prototypes, final_bg_prototypes], axis=0)
    print(f"最終全局原型已組合，總形狀: {all_prototypes.shape}")
    
    output_path = cfg['output_prototype_path']
    np.save(output_path, all_prototypes)
    print(f"全局原型已成功保存至: {output_path}")

if __name__ == '__main__':
    # --- 配置參數 ---
    prototype_generation_config = {
        'gpu_id': 0,
        'image_dir': '/home/U116med/data/polyp/TrainDataset/images/',
        'mask_dir': '/home/U116med/data/polyp/TrainDataset/masks/',
        'image_size': 352,
        'batch_size': 4,
        
        'target_stage_idx': 1, # 使用 PVTv2 Stage 2 (44x44)
        
        # 階段一: 圖像內 K-Means 參數
        'n_fg_prototypes_per_image': 8,
        'n_bg_prototypes_per_image': 8,

        # 階段二: 全局 K-Means 參數
        'n_fg_prototypes': 8, 
        'n_bg_prototypes': 8,
        
        'output_prototype_path': './global_prototypes_128d.npy',
        'visualization_dir': './prototype_generation_check/'
    }
    
    generate_prototypes(prototype_generation_config)