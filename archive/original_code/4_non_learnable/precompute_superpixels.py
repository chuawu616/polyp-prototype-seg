import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    from skimage.segmentation import slic, mark_boundaries
    from skimage.util import img_as_float
except ImportError:
    raise ImportError('pip install scikit-image')


# ── GT-guided SLIC ─────────────────────────────────────────────────────────────

def compute_gt_slic(img_np, gt_np, pixels_per_sp=50,
                    compactness=10.0, sigma=1.0,
                    min_segments=3):
    """
    在 FG 和 BG 區域分別獨立計算 SLIC，確保沒有 superpixel 跨越 GT 邊界。

    Args:
        img_np:        (H, W, 3) uint8 RGB，已 resize 至訓練尺寸
        gt_np:         (H, W)   uint8，值為 0（BG）或 1（FG）
        pixels_per_sp: 每個 superpixel 的目標 pixel 數
                       小 → 切更細（接近 pixel-level）
                       大 → 切更粗（空間連續性更強）
        compactness:   形狀規則性（高 = 方形，低 = 依顏色）
        sigma:         SLIC 前的 Gaussian 模糊
        min_segments:  每個區域的最少 superpixel 數（防止極小 polyp 崩潰）
    Returns:
        segments:      (H, W) int32，BG=0..n_bg-1，FG=n_bg..n_bg+n_fg-1
        n_bg:          BG superpixel 數
        n_fg:          FG superpixel 數
    """
    H, W = gt_np.shape
    fg_mask = (gt_np == 1)
    bg_mask = (gt_np == 0)

    n_fg_pixels = fg_mask.sum()
    n_bg_pixels = bg_mask.sum()

    # 根據區域面積動態計算 superpixel 數量
    n_fg = max(min_segments, int(n_fg_pixels / pixels_per_sp))
    n_bg = max(min_segments, int(n_bg_pixels / pixels_per_sp))

    img_float = img_as_float(img_np)
    segments  = np.zeros((H, W), dtype=np.int32)

    # ── BG SLIC ────────────────────────────────────────────────────────────
    if n_bg_pixels > 0:
        bg_slic = slic(img_float,
                       n_segments=n_bg,
                       compactness=compactness,
                       sigma=sigma,
                       mask=bg_mask,
                       start_label=0)
        # mask=bg_mask 讓 FG 區域的 index 填 -1（skimage 行為），需要重新 index
        # 只取 BG 區域的值
        bg_ids   = np.unique(bg_slic[bg_mask])
        bg_remap = {old: new for new, old in enumerate(bg_ids)}
        for old, new in bg_remap.items():
            segments[bg_slic == old] = new
        n_bg_actual = len(bg_ids)
    else:
        n_bg_actual = 0

    # ── FG SLIC ────────────────────────────────────────────────────────────
    if n_fg_pixels > 0:
        fg_slic = slic(img_float,
                       n_segments=n_fg,
                       compactness=compactness,
                       sigma=sigma,
                       mask=fg_mask,
                       start_label=0)
        fg_ids   = np.unique(fg_slic[fg_mask])
        fg_remap = {old: new + n_bg_actual for new, old in enumerate(fg_ids)}
        for old, new in fg_remap.items():
            segments[fg_slic == old] = new
        n_fg_actual = len(fg_ids)
    else:
        n_fg_actual = 0

    return segments.astype(np.int32), n_bg_actual, n_fg_actual


# ── Visualization ──────────────────────────────────────────────────────────────

def visualize_gt_slic(img_np, gt_np, segments, n_bg, n_fg, save_path):
    """
    四欄視覺化：原圖 | GT mask | Superpixel 邊界（GT 著色）| FG/BG 分別著色
    """
    H, W = gt_np.shape

    # 隨機著色 superpixel map
    n_total = n_bg + n_fg
    rng     = np.random.default_rng(42)
    palette = rng.integers(0, 256, size=(n_total, 3), dtype=np.uint8)
    colored = palette[segments]   # (H, W, 3)

    # BG/FG 用不同色系
    palette_bg = (rng.random((n_bg, 3)) * 0.5 + 0.5)           # 亮色
    palette_fg = (rng.random((n_fg, 3)) * 0.5)                  # 深色
    class_colored = np.zeros((H, W, 3))
    for i in range(n_bg):
        class_colored[segments == i] = palette_bg[i]
    for i in range(n_fg):
        class_colored[segments == (n_bg + i)] = palette_fg[i]

    # Superpixel 邊界疊加
    boundary_img = mark_boundaries(img_as_float(img_np), segments,
                                   color=(1, 0, 0), mode='thick')

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img_np);         axes[0].set_title('Original')
    axes[1].imshow(gt_np, cmap='gray'); axes[1].set_title(f'GT (FG={gt_np.sum()//1} px)')
    axes[2].imshow(boundary_img);   axes[2].set_title(f'SLIC boundary (BG={n_bg}, FG={n_fg})')
    axes[3].imshow(class_colored);  axes[3].set_title('BG(bright) / FG(dark) superpixels')
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir',      type=str,
                        default='/home/U116med/data/polyp/TrainDataset/images/')
    parser.add_argument('--mask_dir',       type=str,
                        default='/home/U116med/data/polyp/TrainDataset/masks/')
    parser.add_argument('--save_dir',       type=str,
                        default='/home/U116med/wch_code/non_learnable/superpixels/')
    parser.add_argument('--img_size',       type=int, default=352,
                        help='訓練 resize 尺寸（需和 training 一致）')
    parser.add_argument('--pixels_per_sp',  type=int, default=50,
                        help='每個 superpixel 的目標 pixel 數（越小切越細）'
                             '  d1 解析度 88×88：'
                             '  50 → FG 約 2~120 個 sp（隨 polyp 大小縮放）')
    parser.add_argument('--compactness',    type=float, default=10.0)
    parser.add_argument('--sigma',          type=float, default=1.0)
    parser.add_argument('--min_segments',   type=int, default=3,
                        help='每個區域最少 superpixel 數')
    parser.add_argument('--n_vis',          type=int, default=10,
                        help='儲存視覺化範例數（0 = 不儲存）')
    parser.add_argument('--vis_dir',        type=str,
                        default='./superpixel_gt_vis/')
    opt = parser.parse_args()

    os.makedirs(opt.save_dir, exist_ok=True)
    if opt.n_vis > 0:
        os.makedirs(opt.vis_dir, exist_ok=True)

    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'}
    img_names = sorted([
        f for f in os.listdir(opt.image_dir)
        if os.path.splitext(f)[1].lower() in valid_ext
    ])

    if not img_names:
        print(f'找不到圖片：{opt.image_dir}')
        exit(1)

    print(f'共 {len(img_names)} 張圖片')
    print(f'pixels_per_sp={opt.pixels_per_sp}, '
          f'compactness={opt.compactness}, sigma={opt.sigma}')
    print(f'輸出：{opt.save_dir}')

    n_bg_list, n_fg_list = [], []

    for idx, fname in enumerate(tqdm(img_names)):
        stem     = os.path.splitext(fname)[0]
        img_path = os.path.join(opt.image_dir, fname)

        # GT mask：嘗試同名 .png，再嘗試同名 .jpg
        gt_path = os.path.join(opt.mask_dir, stem + '.png')
        if not os.path.exists(gt_path):
            gt_path = os.path.join(opt.mask_dir, stem + '.jpg')
        if not os.path.exists(gt_path):
            # 找同名的任意副檔名
            candidates = [f for f in os.listdir(opt.mask_dir)
                          if os.path.splitext(f)[0] == stem]
            if candidates:
                gt_path = os.path.join(opt.mask_dir, candidates[0])
            else:
                print(f'Warning: GT not found for {fname}, skipping.')
                continue

        # 讀取並 resize
        img = Image.open(img_path).convert('RGB')
        gt  = Image.open(gt_path).convert('L')
        img = img.resize((opt.img_size, opt.img_size), Image.BILINEAR)
        gt  = gt.resize((opt.img_size, opt.img_size), Image.NEAREST)

        img_np = np.array(img)                              # (H, W, 3) uint8
        gt_np  = (np.array(gt) > 128).astype(np.uint8)     # (H, W) 0/1

        # GT-guided SLIC
        segments, n_bg, n_fg = compute_gt_slic(
            img_np, gt_np,
            pixels_per_sp=opt.pixels_per_sp,
            compactness=opt.compactness,
            sigma=opt.sigma,
            min_segments=opt.min_segments
        )

        n_bg_list.append(n_bg)
        n_fg_list.append(n_fg)

        # 儲存
        np.save(os.path.join(opt.save_dir, f'{stem}.npy'), segments)

        # 視覺化
        if opt.n_vis > 0 and idx < opt.n_vis:
            vis_path = os.path.join(opt.vis_dir, f'{stem}_gt_sp.jpg')
            visualize_gt_slic(img_np, gt_np, segments, n_bg, n_fg, vis_path)

    # 統計摘要
    n_bg_arr = np.array(n_bg_list)
    n_fg_arr = np.array(n_fg_list)
    print(f'\n=== 完成 ===')
    print(f'BG superpixels：mean={n_bg_arr.mean():.1f}  '
          f'min={n_bg_arr.min()}  max={n_bg_arr.max()}')
    print(f'FG superpixels：mean={n_fg_arr.mean():.1f}  '
          f'min={n_fg_arr.min()}  max={n_fg_arr.max()}')
    print(f'Total mean：{(n_bg_arr + n_fg_arr).mean():.1f}')
    print(f'儲存：{opt.save_dir}')