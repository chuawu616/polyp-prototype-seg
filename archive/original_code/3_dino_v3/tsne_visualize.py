import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import random

from lib.networks_prototype_specialize import DINOv3_Prototype_Specialize
from lib.networks_prototype_specialize_pvtv2 import Prototype_CASCADE_Specialize
from utils.dataloader import test_dataset

def main():
    parser = argparse.ArgumentParser(description="t-SNE Visualization for Prototypes and Pixel Features")
    parser.add_argument('--pth_path', type=str, 
                        default='/home/U116med/wch_code/cascade/models/prototype_cascade/baseline_prototype_fhb_424_warm_10_orth_loss/best.pth', 
                        help='訓練好的 Specialize 模型 weights 路徑')
    parser.add_argument('--data_path', type=str, default='/home/U116med/data/polyp/TestDataset/CVC-ColonDB', help='測試資料集路徑')
    parser.add_argument('--output_dir', type=str, default='./analysis_results/', help='圖表儲存目錄')
    parser.add_argument('--model_type', type=str, default='cascade', choices=['dino', 'cascade'], help='使用的模型架構')
    parser.add_argument('--backbone', type=str, default='vits16plus', help='DINOv3 版本 (若使用 dino)')
    parser.add_argument('--num_fg', type=int, default=4, help='前景原型總數量')
    parser.add_argument('--num_bg', type=int, default=4, help='背景原型數量')
    parser.add_argument('--max_images', type=int, default=15, help='提取特徵的最大圖片數量')
    parser.add_argument('--samples_per_image', type=int, default=200, help='每張圖片隨機抽樣的 FG 像素數量')
    opt = parser.parse_args()

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    print(f"正在載入模型: {opt.pth_path}")
    if opt.model_type == 'dino':
        model = DINOv3_Prototype_Specialize(num_fg=opt.num_fg, num_bg=opt.num_bg, backbone_type=opt.backbone).cuda()
    else:
        model = Prototype_CASCADE_Specialize(num_fg=opt.num_fg, num_bg=opt.num_bg).cuda()
    
    checkpoint = torch.load(opt.pth_path, map_location='cuda')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    # --- 1. 設定 Hook 攔截降維前的 Pixel Embeddings ---
    features = {}
    
    def get_dino_feat(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
        
    def get_cascade_feat(name):
        def hook(model, input, output):
            features[name] = output[-1].detach() # CASCADE 的 decoder 回傳 list，取最後一層
        return hook

    if opt.model_type == 'dino' and hasattr(model, 'proto_proj'):
        model.proto_proj.register_forward_hook(get_dino_feat('embed'))
    elif opt.model_type == 'cascade' and hasattr(model, 'decoder'):
        model.decoder.register_forward_hook(get_cascade_feat('embed'))
    else:
        raise ValueError("無法對應模型架構，請確認 hook 的目標層名稱。")

    # --- 2. 提取 Prototype 權重向量 ---
    # t-SNE 必須在相同的 L2 正規化空間中進行對比
    w_easy = F.normalize(model.easy_prototypes, p=2, dim=1).cpu().detach().numpy()
    w_hard = F.normalize(model.hard_prototypes, p=2, dim=1).cpu().detach().numpy()
    
    fg_easy_pixels = []
    fg_hard_pixels = []

    # --- 3. 提取圖片的 Pixel Embeddings ---
    image_root = os.path.join(opt.data_path, 'images/')
    gt_root = os.path.join(opt.data_path, 'masks/')
    test_loader = test_dataset(image_root, gt_root, 352)
    
    print(f"開始從測試集提取特徵 (最多 {opt.max_images} 張圖片)...")
    images_processed = 0
    
    with torch.no_grad():
        for i in tqdm(range(test_loader.size)):
            if images_processed >= opt.max_images:
                break
                
            image_tensor, gt_mask, _ = test_loader.load_data()
            gt_np = np.asarray(gt_mask, np.float32)
            gt_np /= (gt_np.max() + 1e-8)
            gt_binary = np.where(gt_np >= 0.5, 1, 0)
            
            # 如果這張圖片沒有息肉，跳過不處理
            if gt_binary.sum() == 0:
                continue
                
            image_tensor = image_tensor.cuda()
            _ = model(image_tensor) # 觸發 hook
            
            pixel_embeddings = features['embed'] # (1, C, H_feat, W_feat)
            H_feat, W_feat = pixel_embeddings.shape[2:]
            
            # 將特徵正規化
            embeddings_norm = F.normalize(pixel_embeddings, p=2, dim=1)
            
            # 為了對齊特徵圖，將 GT 縮小至特徵圖尺寸
            gt_feat_size = cv2.resize(gt_binary.astype(np.uint8), (W_feat, H_feat), interpolation=cv2.INTER_NEAREST)
            
            # 重新計算門控權重 (避免依賴模型的回傳格式)
            easy_norm = F.normalize(model.easy_prototypes, p=2, dim=1)
            bg_norm = F.normalize(model.bg_prototypes, p=2, dim=1)
            logit_scale = torch.clamp(model.logit_scale.exp(), max=100)
            
            sim_easy = F.conv2d(embeddings_norm, easy_norm.unsqueeze(-1).unsqueeze(-1)) * logit_scale
            sim_bg = F.conv2d(embeddings_norm, bg_norm.unsqueeze(-1).unsqueeze(-1)) * logit_scale
            
            score_easy = torch.logsumexp(sim_easy, dim=1)
            score_bg = torch.logsumexp(sim_bg, dim=1)
            prob_easy = torch.sigmoid(score_easy - score_bg).squeeze().cpu().numpy() # (H_feat, W_feat)
            
            embeddings_np = embeddings_norm.squeeze().cpu().numpy().transpose(1, 2, 0) # (H_feat, W_feat, C)
            
            # 找出屬於 GT 前景的座標
            fg_coords = np.argwhere(gt_feat_size == 1)
            
            # 隨機抽樣，避免記憶體爆炸
            if len(fg_coords) > opt.samples_per_image:
                indices = np.random.choice(len(fg_coords), opt.samples_per_image, replace=False)
                fg_coords = fg_coords[indices]
                
            for y, x in fg_coords:
                feat_vector = embeddings_np[y, x, :]
                gate_val = prob_easy[y, x]
                
                # 根據門控值分類該像素
                if gate_val > 0.5:
                    fg_easy_pixels.append(feat_vector)
                else:
                    fg_hard_pixels.append(feat_vector)
                    
            images_processed += 1

    fg_easy_pixels = np.array(fg_easy_pixels) if len(fg_easy_pixels) > 0 else np.empty((0, w_easy.shape[1]))
    fg_hard_pixels = np.array(fg_hard_pixels) if len(fg_hard_pixels) > 0 else np.empty((0, w_easy.shape[1]))

    print(f"\n特徵提取完成:")
    print(f"- Easy Prototypes: {w_easy.shape[0]} 個")
    print(f"- Hard Prototypes: {w_hard.shape[0]} 個")
    print(f"- Easy FG Pixels (prob > 0.5): {fg_easy_pixels.shape[0]} 個")
    print(f"- Hard FG Pixels (prob <= 0.5): {fg_hard_pixels.shape[0]} 個")

    # --- 4. 準備 t-SNE 資料 ---
    all_features = []
    labels = []
    
    all_features.extend(w_easy)
    labels.extend(['Prototype: Easy'] * len(w_easy))
    
    all_features.extend(w_hard)
    labels.extend(['Prototype: Hard'] * len(w_hard))
    
    if len(fg_easy_pixels) > 0:
        all_features.extend(fg_easy_pixels)
        labels.extend(['Pixel: FG (Easy Assigned)'] * len(fg_easy_pixels))
        
    if len(fg_hard_pixels) > 0:
        all_features.extend(fg_hard_pixels)
        labels.extend(['Pixel: FG (Hard Assigned)'] * len(fg_hard_pixels))
        
    all_features = np.array(all_features)
    
    # 執行降維
    print("正在執行 t-SNE 降維計算 (這可能需要幾分鐘)...")
    # Perplexity 通常設定在 5 到 50 之間，取決於樣本數量
    perplexity_val = min(30, max(5, len(all_features) // 10))
    tsne = TSNE(n_components=2, perplexity=perplexity_val, random_state=42, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(all_features)

    # --- 5. 繪製圖表 ---
    plt.figure(figsize=(12, 10))
    
    # 定義繪圖樣式
    plot_config = {
        'Pixel: FG (Easy Assigned)': {'marker': 'o', 'color': 'lightcoral', 'alpha': 0.4, 's': 20},
        'Pixel: FG (Hard Assigned)': {'marker': 'o', 'color': 'skyblue', 'alpha': 0.6, 's': 20},
        'Prototype: Easy': {'marker': '*', 'color': 'darkred', 'alpha': 1.0, 's': 400, 'edgecolor': 'black'},
        'Prototype: Hard': {'marker': '*', 'color': 'darkblue', 'alpha': 1.0, 's': 400, 'edgecolor': 'black'}
    }

    # 按照順序繪圖，確保 Prototype 畫在最上層
    for label_name in ['Pixel: FG (Easy Assigned)', 'Pixel: FG (Hard Assigned)', 'Prototype: Easy', 'Prototype: Hard']:
        idx = [i for i, l in enumerate(labels) if l == label_name]
        if len(idx) == 0:
            continue
            
        points = tsne_results[idx]
        cfg = plot_config[label_name]
        
        plt.scatter(points[:, 0], points[:, 1], 
                    label=label_name, 
                    marker=cfg['marker'], 
                    c=cfg['color'], 
                    alpha=cfg['alpha'], 
                    s=cfg['s'], 
                    edgecolors=cfg.get('edgecolor', 'none'))

    plt.title("t-SNE Embedding Space: Prototypes vs FG Pixels", fontsize=16, pad=15)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    save_name = 'fhb_424_warm_10_orth_CVC_ColonDB_tsne_space.png'
    save_path = os.path.join(opt.output_dir, save_name)
    plt.savefig(save_path, dpi=300)
    print(f"t-SNE 圖表已儲存至: {save_path}")

if __name__ == '__main__':
    # 因涉及 cv2，請確保已 import cv2 或在此補充
    import cv2
    main()