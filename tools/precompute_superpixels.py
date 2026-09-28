"""GT-guided SLIC superpixels for the pseudo-label + superpixel experiment.

SLIC runs separately inside the foreground and the background mask, so no superpixel crosses the GT
boundary (plain RGB SLIC would cap the attainable Dice). The number of superpixels scales with region
area: n_segments = area / pixels_per_sp.

  python tools/precompute_superpixels.py --train_root data/polyp/TrainDataset \
      --save_dir data/polyp/TrainDataset/superpixels_gt [--n_vis 10]
"""
import argparse
import os

import matplotlib
import numpy as np
from PIL import Image
from skimage.segmentation import mark_boundaries, slic
from skimage.util import img_as_float
from tqdm import tqdm

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def gt_guided_slic(img, gt, pixels_per_sp=50, compactness=10.0, sigma=1.0, min_segments=3):
    """img (H,W,3) uint8, gt (H,W) {0,1} -> segments (H,W) int32 with BG ids first, then FG ids."""
    segments = np.zeros(gt.shape, dtype=np.int32)
    offset = 0
    for mask in (gt == 0, gt == 1):
        if mask.sum() == 0:
            continue
        n = max(min_segments, int(mask.sum() / pixels_per_sp))
        lab = slic(img_as_float(img), n_segments=n, compactness=compactness, sigma=sigma, mask=mask, start_label=0)
        for new, old in enumerate(np.unique(lab[mask])):
            segments[(lab == old) & mask] = new + offset
        offset += len(np.unique(lab[mask]))
    return segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_root', default='data/polyp/TrainDataset')
    ap.add_argument('--save_dir', default='data/polyp/TrainDataset/superpixels_gt')
    ap.add_argument('--img_size', type=int, default=352, help='must match training resolution')
    ap.add_argument('--pixels_per_sp', type=int, default=50,
                    help='target pixels per superpixel at --img_size; the original runs used 50 at 352, which is '
                         '~3 pixels at the 88x88 d1 level (~800 gives the intended ~50 d1 pixels)')
    ap.add_argument('--compactness', type=float, default=10.0)
    ap.add_argument('--sigma', type=float, default=1.0)
    ap.add_argument('--n_vis', type=int, default=0, help='save this many visualisations to <save_dir>_vis')
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    img_dir, mask_dir = os.path.join(args.train_root, 'images'), os.path.join(args.train_root, 'masks')
    names = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png')))
    for i, fname in enumerate(tqdm(names)):
        stem = os.path.splitext(fname)[0]
        size = (args.img_size, args.img_size)
        img = np.array(Image.open(os.path.join(img_dir, fname)).convert('RGB').resize(size, Image.BILINEAR))
        gt = (np.array(Image.open(os.path.join(mask_dir, stem + '.png')).convert('L').resize(size, Image.NEAREST))
              > 128).astype(np.uint8)
        seg = gt_guided_slic(img, gt, args.pixels_per_sp, args.compactness, args.sigma)
        np.save(os.path.join(args.save_dir, stem + '.npy'), seg)
        if i < args.n_vis:
            os.makedirs(args.save_dir + '_vis', exist_ok=True)
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(img); ax[1].imshow(gt, cmap='gray')
            ax[2].imshow(mark_boundaries(img_as_float(img), seg, color=(1, 0, 0)))
            for a in ax:
                a.axis('off')
            plt.savefig(os.path.join(args.save_dir + '_vis', stem + '.jpg'), dpi=100, bbox_inches='tight')
            plt.close()


if __name__ == '__main__':
    main()
