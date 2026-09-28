# generate_aligned_sublabels.py
# 描述: 基於中心點重映射的、健壯且可續算的流程，生成全域一致的子標籤（sublabels）

import os
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import cv2
from torch.utils.data import Dataset, DataLoader
from models.pvtv2 import pvt_v2_b2


# -------------------------
# Dataset (embedded)
# -------------------------
class FeatureGenDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(352, 352)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        # collect images
        self.image_paths = sorted([p for p in glob.glob(os.path.join(image_dir, '*')) 
                                   if p.lower().endswith(('.png', '.jpg', '.jpeg'))])

        # map base_name -> mask_path (only png masks assumed)
        self.mask_map = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(mask_dir, '*'))
            if os.path.splitext(p)[1].lower() in ('.png', '.jpg', '.jpeg')
        }

        # keep only those with masks
        self.image_paths = [p for p in self.image_paths if os.path.splitext(os.path.basename(p))[0] in self.mask_map]
        if not self.image_paths:
            raise FileNotFoundError("找不到任何匹配的 影像-遮罩 對。")

        print(f"[Dataset] 找到 {len(self.image_paths)} 組影像-遮罩對。")

        self.transform = T.Compose([
            T.Resize(self.image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = self.mask_map[base_name]

        img_pil = Image.open(img_path).convert("RGB")
        mask_pil = Image.open(mask_path).convert("L")

        image_tensor = self.transform(img_pil)

        # NOTE: keep original mask size as np array (we will resize when needed)
        mask_np = np.array(mask_pil)
        mask_binary = (mask_np > 128).astype(np.uint8)

        return image_tensor, mask_binary, img_path


# -------------------------
# Stage 1: generate local centroids + temp sublabels
# -------------------------
def stage1_generate_local_protos_and_labels(cfg, model):
    print("\n--- 阶段一: 生成临时子标签和局部原型 ---")
    device = next(model.parameters()).device

    temp_sublabel_dir = cfg['temp_sublabel_dir']
    local_centroids_dir = cfg['local_centroids_dir']
    os.makedirs(temp_sublabel_dir, exist_ok=True)
    os.makedirs(local_centroids_dir, exist_ok=True)

    dataset = FeatureGenDataset(image_dir=cfg['image_dir'], mask_dir=cfg['mask_dir'], image_size=(cfg['image_size'], cfg['image_size']))
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], num_workers=4, shuffle=False)

    with torch.no_grad():
        for images, masks_np_list, paths in tqdm(dataloader, desc="阶段一"):
            images = images.to(device)  # (B, 3, H, W)

            # 提取 features（假設 forward_features 返回 list 或 tuple）
            features_list = model.forward_features(images)
            feat_map_batch = features_list[cfg['target_stage_idx']]  # (B, C, h, w)

            B = images.shape[0]
            for i in range(B):
                base_name = os.path.splitext(os.path.basename(paths[i]))[0]
                temp_label_path = os.path.join(temp_sublabel_dir, f"{base_name}.png")
                local_centroid_path = os.path.join(local_centroids_dir, f"{base_name}.npy")

                if os.path.exists(temp_label_path) and os.path.exists(local_centroid_path) and not cfg.get('overwrite_stage1', False):
                    continue

                feat_map_tensor = feat_map_batch[i]  # (C, h, w) on GPU
                mask_np_orig = masks_np_list[i]  # (H_orig, W_orig) on CPU numpy uint8

                # create mask tensor (B=1) and resize to feature spatial size using nearest
                # 如果是 numpy → 轉成 tensor
                if isinstance(mask_np_orig, np.ndarray):
                    mask_tensor = torch.from_numpy(mask_np_orig.astype(np.uint8))
                else:
                    # 已經是 tensor → 確保 uint8
                    mask_tensor = mask_np_orig.to(torch.uint8)

                mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).float().to(device)

                mask_small_tensor = F.interpolate(mask_tensor, size=feat_map_tensor.shape[-2:], mode='nearest').squeeze().long()  # (h, w)

                # move feature to CPU for sklearn KMeans (avoid GPU→CPU repeated many times)
                feat_map_hwc = feat_map_tensor.permute(1, 2, 0).cpu().numpy()  # (h, w, C)
                true_mask_small = mask_small_tensor.cpu().numpy().astype(np.uint8)  # (h, w)

                Hs, Ws, C = feat_map_hwc.shape
                pixels_flat = feat_map_hwc.reshape(-1, C)
                mask_flat = true_mask_small.flatten()

                # split fg/bg
                fg_pixels = pixels_flat[mask_flat == 1]
                bg_pixels = pixels_flat[mask_flat == 0]

                temp_sublabel_map = np.zeros_like(true_mask_small, dtype=np.uint8)
                fg_centroids = None
                bg_centroids = None

                # FG clustering
                if fg_pixels.shape[0] >= cfg['n_fg_clusters']:
                    kmeans_fg = KMeans(n_clusters=cfg['n_fg_clusters'], random_state=42, n_init=10)
                    fg_labels = kmeans_fg.fit_predict(fg_pixels)
                    fg_centroids = kmeans_fg.cluster_centers_.astype(np.float32)
                    temp_sublabel_map[true_mask_small == 1] = fg_labels + 1  # 1..n_fg
                else:
                    # if not enough fg pixels, leave them labeled as 1 (single class)
                    if fg_pixels.shape[0] > 0:
                        temp_sublabel_map[true_mask_small == 1] = 1
                        fg_centroids = np.zeros((cfg['n_fg_clusters'], C), dtype=np.float32)

                # BG clustering
                if bg_pixels.shape[0] >= cfg['n_bg_clusters']:
                    kmeans_bg = KMeans(n_clusters=cfg['n_bg_clusters'], random_state=42, n_init=10)
                    bg_labels = kmeans_bg.fit_predict(bg_pixels)
                    bg_centroids = kmeans_bg.cluster_centers_.astype(np.float32)
                    temp_sublabel_map[true_mask_small == 0] = bg_labels + 1 + cfg['n_fg_clusters']  # offset
                else:
                    if bg_pixels.shape[0] > 0:
                        temp_sublabel_map[true_mask_small == 0] = cfg['n_fg_clusters'] + 1
                        bg_centroids = np.zeros((cfg['n_bg_clusters'], C), dtype=np.float32)

                # save temp_sublabel_map as uint8
                temp_sublabel_map_uint8 = temp_sublabel_map.astype(np.uint8)
                Image.fromarray(temp_sublabel_map_uint8).save(temp_label_path)

                # prepare local prototypes (concatenate fg then bg)
                if fg_centroids is None:
                    fg_centroids = np.zeros((cfg['n_fg_clusters'], C), dtype=np.float32)
                if bg_centroids is None:
                    bg_centroids = np.zeros((cfg['n_bg_clusters'], C), dtype=np.float32)

                local_prototypes = np.concatenate([fg_centroids, bg_centroids], axis=0)  # (n_fg + n_bg, C)
                np.save(local_centroid_path, local_prototypes.astype(np.float32))

    print("[Stage1] 完成。暫存子標籤與局部原型已儲存。")


# -------------------------
# Stage 2: calculate global prototypes
# -------------------------
def stage2_calculate_global_prototypes(cfg):
    print("\n--- 阶段二: 计算全局锚点原型 ---")
    local_centroids_dir = cfg['local_centroids_dir']
    local_files = sorted(glob.glob(os.path.join(local_centroids_dir, '*.npy')))
    if not local_files:
        raise FileNotFoundError(f"在 {local_centroids_dir} 中找不到任何局部中心文件。")

    all_fg, all_bg = [], []
    num_fg = cfg['n_fg_clusters']
    num_bg = cfg['n_bg_clusters']

    for p in tqdm(local_files, desc="加载局部中心"):
        arr = np.load(p)  # (n_fg + n_bg, C)
        all_fg.append(arr[:num_fg])
        all_bg.append(arr[num_fg:])

    fg_pool = np.concatenate(all_fg, axis=0)
    bg_pool = np.concatenate(all_bg, axis=0)

    # 全局 KMeans
    kmeans_fg = KMeans(n_clusters=num_fg, random_state=42, n_init=20).fit(fg_pool)
    global_fg = kmeans_fg.cluster_centers_.astype(np.float32)

    kmeans_bg = KMeans(n_clusters=num_bg, random_state=42, n_init=20).fit(bg_pool)
    global_bg = kmeans_bg.cluster_centers_.astype(np.float32)

    print("[Stage2] 全局原型計算完成。")
    return global_fg, global_bg


# -------------------------
# Stage 3: remap local labels -> global consistent labels
# -------------------------
def stage3_remap_labels(cfg, global_fg_prototypes, global_bg_prototypes):
    print("\n--- 阶段三: 重映射临时标签為全域一致標籤 ---")
    local_centroids_dir = cfg['local_centroids_dir']
    temp_sublabel_dir = cfg['temp_sublabel_dir']
    final_sublabel_dir = cfg['final_sublabel_dir']
    os.makedirs(final_sublabel_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(local_centroids_dir, '*.npy')))
    for local_centroid_path in tqdm(files, desc="阶段三"):
        base_name = os.path.splitext(os.path.basename(local_centroid_path))[0]
        temp_label_path = os.path.join(temp_sublabel_dir, f"{base_name}.png")
        final_path = os.path.join(final_sublabel_dir, f"{base_name}.png")

        if not os.path.exists(temp_label_path):
            continue
        if os.path.exists(final_path) and not cfg.get('overwrite_stage3', False):
            continue

        local_prototypes = np.load(local_centroid_path)  # (n_fg + n_bg, C)
        num_fg = cfg['n_fg_clusters']

        local_fg = local_prototypes[:num_fg]
        local_bg = local_prototypes[num_fg:]

        temp_map = np.array(Image.open(temp_label_path)).astype(np.int32)
        final_map = np.zeros_like(temp_map, dtype=np.uint8)

        # remap fg
        if np.any(local_fg):
            sims_fg = cosine_similarity(local_fg, global_fg_prototypes)  # (local_fg_count, global_fg_count)
            remap_fg = np.argmax(sims_fg, axis=1)  # which global FG prototype each local fg maps to
            for local_id in range(local_fg.shape[0]):
                final_map[temp_map == (local_id + 1)] = remap_fg[local_id] + 1

        # remap bg (offset by num_fg)
        if np.any(local_bg):
            sims_bg = cosine_similarity(local_bg, global_bg_prototypes)
            remap_bg = np.argmax(sims_bg, axis=1)
            for local_id in range(local_bg.shape[0]):
                final_map[temp_map == (local_id + 1 + num_fg)] = remap_bg[local_id] + 1 + num_fg

        # resize to desired image_size (nearest)
        final_map_resized = cv2.resize(final_map, (cfg['image_size'], cfg['image_size']), interpolation=cv2.INTER_NEAREST)

        Image.fromarray(final_map_resized.astype(np.uint8)).save(final_path)

    print("[Stage3] 完成。全域一致的子標籤已儲存。")


# -------------------------
# Main
# -------------------------
if __name__ == '__main__':
    config = {
        'image_dir': '/home/U116med/data/polyp/TrainDataset/images/',
        'mask_dir': '/home/U116med/data/polyp/TrainDataset/masks/',
        'stage2_pretrained_path': './pretrained_stage2_weighted/stage2_epoch_20.pth',
        'image_size': 352,
        'batch_size': 1,
        'target_stage_idx': 1,
        'n_fg_clusters': 4,
        'n_bg_clusters': 4,
        'temp_sublabel_dir': './polypdata/sublabels_temp/',
        'local_centroids_dir': './polypdata/centroids_local/',
        'final_sublabel_dir': './polypdata/sublabels_consistent_k8/',
        'overwrite_stage1': False,
        'overwrite_stage3': False,
    }

    print("[Init] 準備中...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pvt_v2_b2(pretrained=False).to(device).eval()

    # 嘗試載入 stage2 權重（若有）
    try:
        state_dict = torch.load(config['stage2_pretrained_path'], map_location=device)
        model.load_state_dict(state_dict)
        print(f"[Init] 成功載入 Stage2 權重: {config['stage2_pretrained_path']}")
    except Exception as e:
        print(f"[Init] 無法載入 Stage2 權重 ({config['stage2_pretrained_path']}): {e}")
        print("[Init] 將使用模型初始化權重執行 (注意: feature distribution 可能與訓練用不同)。")

    # Run pipeline
    stage1_generate_local_protos_and_labels(config, model)
    global_fg_prototypes, global_bg_prototypes = stage2_calculate_global_prototypes(config)
    stage3_remap_labels(config, global_fg_prototypes, global_bg_prototypes)

    print("\n[Done] 所有流程已成功完成！")
