import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from tqdm import tqdm

from lib.networks_prototype_pseudo_v2 import Prototype_Pseudo


# ── Feature Extraction ─────────────────────────────────────────────────────────

def extract_all_node_features(model, images):
    """Backbone + Decoder 所有節點特徵提取。"""
    with torch.no_grad():
        x1, x2, x3, x4 = model.backbone(images)
        dd4, dd3, dd2, dd1, d1 = model.decoder(x4, [x3, x2, x1])
    return {
        'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4,
        'dd4': dd4, 'dd3': dd3, 'dd2': dd2, 'dd1': dd1, 'd1': d1
    }


def apply_pca_to_feature(feat_tensor, target_size=(352, 352)):
    """Feature map → bilinear 上採樣 → PCA 3D → RGB 視覺化。"""
    feat_up = F.interpolate(feat_tensor, size=target_size,
                            mode='bilinear', align_corners=False)
    feat = feat_up.squeeze(0)   # (C, H, W)
    C, H, W = feat.shape

    feat_flat = feat.permute(1, 2, 0).reshape(-1, C).cpu().numpy()
    pca       = PCA(n_components=3)
    feat_pca  = pca.fit_transform(feat_flat)

    feat_pca_norm = np.zeros_like(feat_pca)
    for i in range(3):
        mn, mx = feat_pca[:, i].min(), feat_pca[:, i].max()
        if mx > mn:
            feat_pca_norm[:, i] = (feat_pca[:, i] - mn) / (mx - mn)

    return feat_pca_norm.reshape(H, W, 3)


# ── Image Loading ──────────────────────────────────────────────────────────────

def load_image_and_gt_tensor(img_path, gt_path, img_size=352):
    image = Image.open(img_path).convert('RGB')
    gt    = Image.open(gt_path).convert('L')

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img_tensor  = transform(image).unsqueeze(0).cuda()
    img_vis     = np.array(image.resize((img_size, img_size)))
    img_vis_cv2 = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
    gt_vis      = np.array(gt.resize((img_size, img_size)))
    gt_vis      = np.where(gt_vis > 128, 1, 0).astype(np.uint8)
    return img_tensor, img_vis_cv2, gt_vis


# ── Four-Panel Visualization ───────────────────────────────────────────────────

def get_pseudo_evolution_img(original_image, true_mask, binary_pred,
                             k_class_pred, epoch, num_prototype):
    """
    四欄圖：Original（含 Epoch 標籤）| GT | Binary Pred | K-Class Prototypes

    Prototype_Pseudo channel 排列：
      BG prototypes：index 0 .. M-1
      FG prototypes：index M .. 2M-1
    """
    h, w, _ = original_image.shape

    def _resize(m):
        if m.shape != (h, w):
            return cv2.resize(m.astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST)
        return m

    true_mask    = _resize(true_mask)
    binary_pred  = _resize(binary_pred)
    k_class_pred = _resize(k_class_pred)

    font = cv2.FONT_HERSHEY_SIMPLEX

    def add_title(img, txt, text_color=(0, 0, 0)):
        img = cv2.copyMakeBorder(img, 40, 0, 0, 0,
                                 cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.putText(img, txt, (10, 30), font, 0.7, text_color, 2, cv2.LINE_AA)
        return img

    # [A] Original
    vis_orig = add_title(original_image.copy(), f'Epoch {epoch} | Original')

    # [B] Ground Truth（黃色）
    vis_gt = original_image.copy()
    ov     = vis_gt.copy()
    ov[true_mask == 1] = [0, 255, 255]
    vis_gt = cv2.addWeighted(ov, 0.5, vis_gt, 0.5, 0)
    vis_gt = add_title(vis_gt, 'Ground Truth')

    # [C] Binary Prediction（綠色）
    # Prototype_Pseudo 的 binary pred 已是 softmax 機率，不需要 sigmoid
    vis_bin = original_image.copy()
    ov      = vis_bin.copy()
    ov[binary_pred == 1] = [0, 255, 0]
    vis_bin = cv2.addWeighted(ov, 0.5, vis_bin, 0.5, 0)
    vis_bin = add_title(vis_bin, 'Binary Pred', text_color=(0, 0, 255))

    # [D] K-Class Prototype Map
    # BG 亮色（index 0..M-1），FG 深色（index M..2M-1）
    M = num_prototype

    fg_palette = np.array([
        [0, 255, 0],    [255, 100, 255], [0, 255, 255],  [255, 255, 0],
        [255, 150, 0],  [255, 50, 50],   [100, 200, 255],[255, 255, 255],
        [150, 255, 0],  [0, 255, 150],   [200, 100, 255],[255, 0, 150],
        [100, 255, 200],[255, 200, 100], [50, 150, 255], [200, 255, 100],
        [180, 180, 255],[255, 180, 180], [180, 255, 180],[220, 220, 220]
    ], dtype=np.uint8)

    bg_palette = np.array([
        [0, 0, 80],   [50, 0, 0],   [0, 50, 0],   [50, 50, 0],
        [50, 0, 50],  [0, 50, 50],  [30, 30, 30], [0, 0, 0],
        [0, 0, 50],   [30, 0, 0],   [0, 30, 0],   [40, 40, 40],
        [20, 20, 60], [60, 20, 20], [20, 60, 20], [60, 60, 0],
        [40, 0, 80],  [80, 0, 40],  [0, 80, 40],  [20, 20, 20]
    ], dtype=np.uint8)

    k_colored = np.zeros((h, w, 3), dtype=np.uint8)
    for bg_id in range(min(len(bg_palette), M)):
        k_colored[k_class_pred == bg_id]       = bg_palette[bg_id]
    for fg_id in range(min(len(fg_palette), M)):
        k_colored[k_class_pred == (M + fg_id)] = fg_palette[fg_id]

    vis_k = add_title(k_colored, f'K-Class Prototypes (M={M})')

    combined = np.concatenate((vis_orig, vis_gt, vis_bin, vis_k), axis=1)
    return cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size',      type=int,  default=352)
    parser.add_argument('--model_dir',     type=str,
                        default='models/prototype_pseudo/prototype_pseudo_fb33_sp')
    parser.add_argument('--img_path',      type=str,
                        default='/home/U116med/data/polyp/TrainDataset/images/20.png')
    parser.add_argument('--gt_path',       type=str,
                        default='/home/U116med/data/polyp/TrainDataset/masks/20.png')
    parser.add_argument('--save_dir',      type=str,
                        default='./evolution_pseudo_fb33_sp/')
    parser.add_argument('--num_prototype', type=int,  default=3)
    parser.add_argument('--epochs',        type=int, nargs='+',
                        default=[1, 2, 3, 4, 5, 10, 20, 50, 100],
                        help='要視覺化的 epoch 清單（對應 model_dir/epoch_N.pth）')
    opt = parser.parse_args()

    os.makedirs(opt.save_dir, exist_ok=True)
    img_name = os.path.basename(opt.img_path).split('.')[0]

    # d1 是 prototype 的作用節點，標紅色
    nodes_to_visualize = ['dd1', 'dd2', 'dd3', 'dd4', 'd1', 'x1', 'x2', 'x3', 'x4']
    num_cols     = 1 + len(nodes_to_visualize)
    width_ratios = [4] + [1] * len(nodes_to_visualize)

    # 過濾掉不存在的 epoch weight，避免空行
    valid_epochs = []
    for e in opt.epochs:
        wp = os.path.join(opt.model_dir, f'epoch_{e}.pth')
        if os.path.exists(wp):
            valid_epochs.append(e)
        else:
            print(f'Warning: epoch_{e}.pth not found, skipping.')

    if not valid_epochs:
        print('No valid epoch weights found. Exiting.')
        exit(1)

    fig, axes = plt.subplots(
        len(valid_epochs), num_cols,
        figsize=(4 * num_cols + 12, 4 * len(valid_epochs)),
        gridspec_kw={'width_ratios': width_ratios}
    )
    if len(valid_epochs) == 1:
        axes = np.expand_dims(axes, axis=0)

    img_tensor, img_vis_cv2, gt_vis = load_image_and_gt_tensor(
        opt.img_path, opt.gt_path, opt.img_size
    )

    print(f'Evolution Analysis for: {img_name}')

    for row_idx, epoch in enumerate(tqdm(valid_epochs)):
        weight_path = os.path.join(opt.model_dir, f'epoch_{epoch}.pth')

        # ── Model Loading ──────────────────────────────────────────────────
        # pretrained_model_path=weight_path：
        #   - 跳過 __init__ 裡的 ImageNet backbone 載入
        #   - 改由 _load_pretrained_model 載入完整 epoch weight（含 backbone）
        # 這樣對每個 epoch 都能正確還原當時的全部權重，且不做重複載入
        model = Prototype_Pseudo(
            num_classes=2,
            num_prototype=opt.num_prototype,
            pretrained_model_path=weight_path   # 直接載入 epoch weight
        ).cuda()
        model.eval()

        with torch.no_grad():
            # Prototype_Pseudo eval 輸出：
            #   pred_binary_high: (B, 2, H, W) softmax 機率
            #   similarity_map_high: (B, K*M, H, W) prototype cosine 相似度
            pred_binary_high, sim_map_high = model(img_tensor)

            # Binary prediction：FG channel 機率（已是 [0,1]，不需 sigmoid）
            fg_prob = pred_binary_high[:, 1, :, :].squeeze().cpu().numpy()
            fg_prob = (fg_prob - fg_prob.min()) / (fg_prob.max() - fg_prob.min() + 1e-8)
            binary_pred = np.where(fg_prob >= 0.5, 1, 0).astype(np.uint8)

            # K-Class prototype assignment（similarity_map argmax）
            k_class_pred = sim_map_high.squeeze(0).argmax(dim=0).cpu().numpy()

            # Node features for PCA
            node_features = extract_all_node_features(model, img_tensor)

        # [A] 四欄主圖
        four_panel = get_pseudo_evolution_img(
            original_image=img_vis_cv2,
            true_mask=gt_vis,
            binary_pred=binary_pred,
            k_class_pred=k_class_pred,
            epoch=epoch,
            num_prototype=opt.num_prototype
        )
        ax_main = axes[row_idx, 0]
        ax_main.imshow(four_panel)
        ax_main.axis('off')

        # [B] 各節點 PCA
        for col_idx, node in enumerate(nodes_to_visualize):
            feat_rgb = apply_pca_to_feature(
                node_features[node],
                target_size=(opt.img_size, opt.img_size)
            )
            ax_pca = axes[row_idx, 1 + col_idx]
            # d1 是 prototype 節點，標紅色
            title_color = 'red' if node == 'd1' else 'black'
            ax_pca.imshow(feat_rgb)
            ax_pca.set_title(f'PCA: {node.upper()}', fontsize=14,
                             color=title_color)
            ax_pca.axis('off')

    plt.tight_layout()
    save_path = os.path.join(opt.save_dir, f'{img_name}_pseudo_evolution.jpg')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

    print(f'\nSaved to: {save_path}')
