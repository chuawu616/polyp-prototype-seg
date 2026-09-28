import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

from lib.networks_prototype_pseudo_v2 import Prototype_Pseudo
from utils.dataloader_pseudo import test_dataset


# ── Feature Extraction ─────────────────────────────────────────────────────────

def extract_all_node_features(model, images):
    """backbone + decoder 所有節點特徵提取（與 v0 腳本相同）。"""
    with torch.no_grad():
        x1, x2, x3, x4 = model.backbone(images)
        dd4, dd3, dd2, dd1, d1 = model.decoder(x4, [x3, x2, x1])
    return {
        'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4,
        'dd4': dd4, 'dd3': dd3, 'dd2': dd2, 'dd1': dd1, 'd1': d1
    }


def apply_pca_to_feature(feat_tensor, target_size=(352, 352)):
    """Feature map → PCA 3D → RGB 視覺化。"""
    feat = feat_tensor.squeeze(0)   # (C, H, W)
    C, H, W = feat.shape

    feat_flat = feat.permute(1, 2, 0).reshape(-1, C).cpu().numpy()

    pca = PCA(n_components=3)
    feat_pca = pca.fit_transform(feat_flat)

    feat_pca_norm = np.zeros_like(feat_pca)
    for i in range(3):
        mn, mx = feat_pca[:, i].min(), feat_pca[:, i].max()
        if mx > mn:
            feat_pca_norm[:, i] = (feat_pca[:, i] - mn) / (mx - mn)

    feat_rgb = feat_pca_norm.reshape(H, W, 3)
    feat_rgb_t = torch.from_numpy(feat_rgb).permute(2, 0, 1).unsqueeze(0).float()
    feat_rgb_up = F.interpolate(feat_rgb_t, size=target_size, mode='bilinear', align_corners=False)
    return feat_rgb_up.squeeze(0).permute(1, 2, 0).numpy()


# ── Image / GT Loading ─────────────────────────────────────────────────────────

def load_image_and_gt_tensor(img_path, gt_path, img_size=352):
    image = Image.open(img_path).convert('RGB')
    gt    = Image.open(gt_path).convert('L')

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).cuda()
    img_vis    = np.array(image.resize((img_size, img_size)))
    img_vis_cv2 = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
    gt_vis = np.array(gt.resize((img_size, img_size)))
    gt_vis = np.where(gt_vis > 128, 1, 0).astype(np.uint8)
    return img_tensor, img_vis_cv2, gt_vis


# ── Four-Panel Visualization ───────────────────────────────────────────────────

def get_pseudo_case_img(original_image, true_mask, binary_pred,
                        k_class_pred, dice_score, num_prototype):
    """
    四欄圖：Original | Ground Truth | Binary Pred (DICE) | K-Class Prototypes

    Prototype_Pseudo 特別說明：
    - binary_pred：來自 seg_binary（softmax 機率 → 閾值 0.5）
    - k_class_pred：來自 similarity_map argmax（prototype cosine 相似度）
    """
    h, w, _ = original_image.shape

    def _resize(m):
        if m.shape != (h, w):
            return cv2.resize(m.astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST)
        return m

    true_mask   = _resize(true_mask)
    binary_pred = _resize(binary_pred)
    k_class_pred = _resize(k_class_pred)

    font = cv2.FONT_HERSHEY_SIMPLEX

    def add_title(img, txt, text_color=(0, 0, 0)):
        img = cv2.copyMakeBorder(img, 40, 0, 0, 0,
                                 cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.putText(img, txt, (10, 30), font, 0.7, text_color, 2, cv2.LINE_AA)
        return img

    # [A] Original
    vis_orig = add_title(original_image.copy(), "Original")

    # [B] Ground Truth（黃色）
    vis_gt = original_image.copy()
    ov = vis_gt.copy()
    ov[true_mask == 1] = [0, 255, 255]
    vis_gt = cv2.addWeighted(ov, 0.5, vis_gt, 0.5, 0)
    vis_gt = add_title(vis_gt, "Ground Truth")

    # [C] Binary Prediction（綠色）+ DICE
    vis_bin = original_image.copy()
    ov = vis_bin.copy()
    ov[binary_pred == 1] = [0, 255, 0]
    vis_bin = cv2.addWeighted(ov, 0.5, vis_bin, 0.5, 0)
    vis_bin = add_title(vis_bin, f"Binary Pred (DICE: {dice_score:.4f})",
                        text_color=(0, 0, 255))

    # [D] K-Class Prototype map（BG 深色、FG 亮色）
    fg_palette = np.array([
        [0, 255, 0],   [255, 100, 255], [0, 255, 255],  [255, 255, 0],
        [255, 150, 0], [255, 50, 50],   [100, 200, 255], [255, 255, 255],
        [150, 255, 0], [0, 255, 150],   [200, 100, 255], [255, 0, 150],
        [100, 255, 200],[255, 200, 100],[50, 150, 255],  [200, 255, 100],
        [180, 180, 255],[255, 180, 180],[180, 255, 180], [220, 220, 220]
    ], dtype=np.uint8)

    bg_palette = np.array([
        [0, 0, 80],    [50, 0, 0],    [0, 50, 0],   [50, 50, 0],
        [50, 0, 50],   [0, 50, 50],   [30, 30, 30], [0, 0, 0],
        [0, 0, 50],    [30, 0, 0],    [0, 30, 0],   [40, 40, 40],
        [20, 20, 60],  [60, 20, 20],  [20, 60, 20], [60, 60, 0],
        [40, 0, 80],   [80, 0, 40],   [0, 80, 40],  [20, 20, 20]
    ], dtype=np.uint8)

    M = num_prototype
    k_colored = np.zeros((h, w, 3), dtype=np.uint8)
    # BG prototypes: indices 0..M-1
    for bg_id in range(min(len(bg_palette), M)):
        k_colored[k_class_pred == bg_id] = bg_palette[bg_id]
    # FG prototypes: indices M..2M-1
    for fg_id in range(min(len(fg_palette), M)):
        k_colored[k_class_pred == (M + fg_id)] = fg_palette[fg_id]

    vis_k = add_title(k_colored, f"K-Class Prototypes (M={M})")

    combined = np.concatenate((vis_orig, vis_gt, vis_bin, vis_k), axis=1)
    return cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize',      type=int,  default=352)
    parser.add_argument('--pth_path',      type=str,
                        default='models/prototype_pseudo/prototype_pseudo_fb33_sp/best.pth')
    parser.add_argument('--dataset',       type=str,  default='test')
    parser.add_argument('--worst_k',       type=int,  default=30,
                        help='取最差（或最佳）的 k 張')
    parser.add_argument('--save_dir',      type=str,
                        default='./analysis_pseudo_fb33_sp/')
    parser.add_argument('--num_prototype', type=int,  default=3)
    parser.add_argument('--best',          type=str,  default='True',
                        help='True = 取最佳 k 張；False = 取最差 k 張')
    opt = parser.parse_args()

    opt.best = opt.best.lower() in ('true', 'yes', '1')
    os.makedirs(opt.save_dir, exist_ok=True)

    # ── Model Loading ──────────────────────────────────────────────────────
    model = Prototype_Pseudo(num_classes=2, num_prototype=opt.num_prototype).cuda()
    model.load_state_dict(torch.load(opt.pth_path, map_location='cpu'))
    model.eval()

    root_path  = '/home/U116med/data/polyp/TestDataset/'
    data_path  = os.path.join(root_path, opt.dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root    = '{}/masks/'.format(data_path)

    print(f'Stage 1: Evaluating {opt.dataset} to find the '
          f'{"best" if opt.best else "worst"} {opt.worst_k} cases...')

    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    num_images  = len(os.listdir(gt_root))
    results_list = []

    # ── Stage 1: DICE for all images ───────────────────────────────────────
    with torch.no_grad():
        for i in tqdm(range(num_images)):
            image_tensor, gt_mask, name = test_loader.load_data()

            gt_np = np.asarray(gt_mask, np.float32)
            gt_np /= (gt_np.max() + 1e-8)
            image_tensor = image_tensor.cuda()

            # Prototype_Pseudo eval output:
            #   pred_binary_high: (B, 2, H, W) softmax 機率（FG = channel 1）
            #   similarity_map_high: (B, K*M, H, W) prototype cosine 相似度
            pred_binary_high, sim_map_high = model(image_tensor)

            # Binary prediction：直接取 FG channel 機率，不需要 sigmoid
            fg_prob = pred_binary_high[:, 1:2, :, :]   # (B, 1, H, W) 已是 [0,1]
            fg_prob = F.interpolate(fg_prob, size=gt_np.shape,
                                    mode='bilinear', align_corners=False)
            fg_np = fg_prob.squeeze().cpu().numpy()
            fg_np = (fg_np - fg_np.min()) / (fg_np.max() - fg_np.min() + 1e-8)
            pred_bin = np.where(fg_np >= 0.5, 1, 0)
            gt_bin   = np.where(gt_np >= 0.5, 1, 0)

            # K-class prototype assignment（similarity_map argmax）
            sim_resized = F.interpolate(sim_map_high, size=gt_np.shape,
                                        mode='bilinear', align_corners=False)
            pred_k = sim_resized.argmax(dim=1).squeeze(0).cpu().numpy()

            smooth = 1
            inter = (pred_bin.reshape(-1) * gt_bin.reshape(-1)).sum()
            dice  = (2 * inter + smooth) / (pred_bin.sum() + gt_bin.sum() + smooth)

            results_list.append({
                'name':     name,
                'dice':     float(dice),
                'img_path': os.path.join(image_root, name),
                'gt_path':  os.path.join(gt_root, name),
            })

    results_list.sort(key=lambda x: x['dice'])
    selected = results_list[-opt.worst_k:] if opt.best else results_list[:opt.worst_k]
    tag = 'best' if opt.best else 'worst'

    record_file = os.path.join(opt.save_dir,
                               f'{tag}_{opt.worst_k}_{opt.dataset}_record.txt')
    with open(record_file, 'w') as f:
        f.write(f'{tag.capitalize()} {opt.worst_k} cases for {opt.dataset}\n')
        f.write('-' * 50 + '\n')
        for rank, item in enumerate(selected):
            line = f'Rank {rank+1:02d} | DICE: {item["dice"]:.4f} | File: {item["name"]}\n'
            print(line.strip())
            f.write(line)

    # ── Stage 2: PCA + Prediction Visualization ────────────────────────────
    print(f'\nStage 2: Generating visualizations for {opt.worst_k} cases...')

    nodes_to_visualize = ['dd1', 'dd2', 'dd3', 'dd4', 'd1', 'x1', 'x2', 'x3', 'x4']
    num_cols     = 1 + len(nodes_to_visualize)
    width_ratios = [4] + [1] * len(nodes_to_visualize)

    fig, axes = plt.subplots(
        opt.worst_k, num_cols,
        figsize=(4 * num_cols + 12, 4 * opt.worst_k),
        gridspec_kw={'width_ratios': width_ratios}
    )
    if opt.worst_k == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx, item in enumerate(tqdm(selected)):
        img_tensor, img_vis_cv2, gt_vis = load_image_and_gt_tensor(
            item['img_path'], item['gt_path'], opt.testsize
        )

        with torch.no_grad():
            # ── Prototype_Pseudo 推論 ──────────────────────────────────────
            pred_binary_high, sim_map_high = model(img_tensor)

            # Binary prediction（softmax FG channel）
            fg_prob = pred_binary_high[:, 1:2, :, :]
            fg_np   = fg_prob.squeeze().cpu().numpy()
            fg_np   = (fg_np - fg_np.min()) / (fg_np.max() - fg_np.min() + 1e-8)
            binary_pred = np.where(fg_np >= 0.5, 1, 0).astype(np.uint8)

            # K-Class prototype assignment（similarity_map argmax）
            k_class_pred = sim_map_high.squeeze(0).argmax(dim=0).cpu().numpy()

            # Node features for PCA
            node_features = extract_all_node_features(model, img_tensor)

        # [A] 四欄主圖
        four_panel = get_pseudo_case_img(
            original_image=img_vis_cv2,
            true_mask=gt_vis,
            binary_pred=binary_pred,
            k_class_pred=k_class_pred,
            dice_score=item['dice'],
            num_prototype=opt.num_prototype
        )
        ax_main = axes[row_idx, 0]
        ax_main.imshow(four_panel)
        ax_main.set_title(
            f'{tag.capitalize()} {row_idx+1:02d} | {item["name"]}',
            fontsize=12, pad=8, loc='left'
        )
        ax_main.axis('off')

        # [B] PCA 各節點
        for col_idx, node in enumerate(nodes_to_visualize):
            feat_rgb = apply_pca_to_feature(node_features[node],
                                            target_size=(opt.testsize, opt.testsize))
            ax_pca = axes[row_idx, 1 + col_idx]
            # d1 是 prototype 作用的節點，標紅色
            title_color = 'red' if node == 'd1' else 'black'
            ax_pca.imshow(feat_rgb)
            ax_pca.set_title(f'PCA: {node.upper()}', fontsize=11,
                             color=title_color)
            ax_pca.axis('off')

    plt.tight_layout()
    fig_path = os.path.join(opt.save_dir,
                            f'{tag}_{opt.worst_k}_{opt.dataset}_pseudo_analysis.jpg')
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f'\nExecution Complete!')
    print(f'Record saved to:        {record_file}')
    print(f'Visualization saved to: {fig_path}')
