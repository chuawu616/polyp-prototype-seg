import os
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np

from lib.networks_prototype_v0 import Prototype_CASCADE_v0


def check_v0_similarities(model_path, num_classes=2, num_prototype=5,
                           save_path='v0_prototype_similarity.png'):
    if not os.path.exists(model_path):
        print(f"找不到模型權重檔: {model_path}")
        return

    print(f"載入模型與權重: {model_path}")
    model = Prototype_CASCADE_v0(num_classes=num_classes, num_prototype=num_prototype)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    print("萃取 Prototype...")
    with torch.no_grad():
        # v0 只有一組 prototype，定義在 d1 特徵空間（64-dim）
        proto = F.normalize(model.prototypes.data, p=2, dim=-1)
        # shape: (num_classes, num_prototype, C) → (num_classes * num_prototype, C)
        proto_flat = proto.view(-1, proto.shape[-1])

        # Cosine 相似度矩陣：(K*M, K*M)
        sim = torch.mm(proto_flat, proto_flat.t()).numpy()

        # 各 class 內部的 intra-class 相似度（off-diagonal）
        intra_sims = {}
        for k in range(num_classes):
            start = k * num_prototype
            end   = start + num_prototype
            block = sim[start:end, start:end]
            mask  = ~np.eye(num_prototype, dtype=bool)
            intra_sims[k] = block[mask]

        # Cross-class 相似度
        cross_sim = sim[:num_prototype, num_prototype:]

    # ── 文字輸出 ───────────────────────────────────────────────────────────
    labels = (
        [f"BG_{i}" for i in range(num_prototype)] +
        [f"FG_{i}" for i in range(num_prototype)]
    )
    pd.options.display.float_format = '{:.3f}'.format
    print("\n======== Prototype Cosine Similarity (d1 layer) ========")
    print(pd.DataFrame(sim, index=labels, columns=labels))

    print("\n======== Intra-class Similarity Summary ========")
    class_names = ['BG', 'FG']
    for k in range(num_classes):
        vals = intra_sims[k]
        print(f"  {class_names[k]}: mean={vals.mean():.3f}  max={vals.max():.3f}  min={vals.min():.3f}")
        print(f"         (理想: 接近 0，表示 prototype 方向分散)")

    print("\n======== Cross-class Similarity Summary ========")
    print(f"  BG × FG: mean={cross_sim.mean():.3f}  max={cross_sim.max():.3f}  min={cross_sim.min():.3f}")
    print(f"           (理想: 接近 -1，表示 BG/FG 完全分離)\n")

    # ── 視覺化 ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(26, 8),
                             gridspec_kw={'width_ratios': [2, 1, 1]})

    # ── 左圖：全矩陣 heatmap ──────────────────────────────────────────────
    ax = axes[0]
    sns.heatmap(
        sim, ax=ax, annot=True, fmt=".2f", cmap="RdBu_r",
        vmin=-1.0, vmax=1.0,
        xticklabels=labels, yticklabels=labels,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    ax.set_title('Prototype Cosine Similarity (d1 layer)', fontsize=15, pad=12)
    ax.tick_params(axis='x', rotation=45)

    # BG / FG 分界線
    ax.axhline(y=num_prototype, color='black', linewidth=2.5)
    ax.axvline(x=num_prototype, color='black', linewidth=2.5)

    # 標注各象限
    for (r, c, txt) in [
        (num_prototype / 2, num_prototype / 2, 'BG intra'),
        (num_prototype + num_prototype / 2, num_prototype + num_prototype / 2, 'FG intra'),
        (num_prototype / 2, num_prototype + num_prototype / 2, 'cross'),
        (num_prototype + num_prototype / 2, num_prototype / 2, 'cross'),
    ]:
        ax.text(c, r, txt, ha='center', va='center',
                fontsize=9, color='black', alpha=0.4,
                fontweight='bold')

    # ── 中圖：Intra-class 分佈（violin）──────────────────────────────────
    ax = axes[1]
    data_violin = []
    for k, name in enumerate(class_names):
        for v in intra_sims[k]:
            data_violin.append({'Class': name, 'Cosine Similarity': v})
    df_violin = pd.DataFrame(data_violin)

    sns.violinplot(data=df_violin, x='Class', y='Cosine Similarity',
                   palette=['steelblue', 'tomato'], ax=ax, inner='box')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title('Intra-class Prototype Similarity\n(off-diagonal, lower = more diverse)',
                 fontsize=13, pad=12)
    ax.set_ylabel('Cosine Similarity')

    # 標注 mean
    for k, name in enumerate(class_names):
        mean_val = intra_sims[k].mean()
        ax.text(k, mean_val + 0.05, f'μ={mean_val:.2f}',
                ha='center', fontsize=10, color='black')

    # ── 右圖：Cross-class 相似度（heatmap）───────────────────────────────
    ax = axes[2]
    cross_labels_row = [f"BG_{i}" for i in range(num_prototype)]
    cross_labels_col = [f"FG_{i}" for i in range(num_prototype)]
    sns.heatmap(
        cross_sim, ax=ax, annot=True, fmt=".2f", cmap="RdBu_r",
        vmin=-1.0, vmax=1.0,
        xticklabels=cross_labels_col,
        yticklabels=cross_labels_row,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    ax.set_title('Cross-class Similarity (BG × FG)\n(lower = better class separation)',
                 fontsize=13, pad=12)
    ax.tick_params(axis='x', rotation=45)

    plt.suptitle(
        f'v0 Prototype Analysis  |  {num_classes} classes × {num_prototype} prototypes  |  d1 (64-dim)',
        fontsize=14, y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"視覺化結果已儲存至: {save_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth_path', type=str,
                        default='models/prototype_cascade/prototype_v0_fb33/best.pth')
    parser.add_argument('--num_prototype', type=int, default=3)
    parser.add_argument('--save_name', type=str,
                        default='v0_prototype_similarity.png')
    opt = parser.parse_args()

    check_v0_similarities(
        model_path=opt.pth_path,
        num_prototype=opt.num_prototype,
        save_path=opt.save_name
    )
