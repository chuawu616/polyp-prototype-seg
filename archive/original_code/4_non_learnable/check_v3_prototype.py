import os
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from lib.networks_prototype_v3 import Prototype_CASCADE_v3

def check_v3_similarities(model_path, target_node='dd2', num_classes=2, num_prototype=5, save_path='v3_sim_comparison.png'):
    if not os.path.exists(model_path):
        print(f"找不到模型權重檔: {model_path}")
        return

    print(f"正在載入模型與權重: {model_path}")
    model = Prototype_CASCADE_v3(num_classes=num_classes, num_prototype=num_prototype, target_node=target_node)
    
    # 載入權重
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    print("萃取 True Prototype 與 Projected Prototype...")
    with torch.no_grad():
        # 1. 取得 True Prototype (確保經過 L2 正規化)
        true_proto = F.normalize(model.prototypes.data, p=2, dim=-1)
        
        # 2. 透過 MLP 產生 Projected Prototype (並經過 L2 正規化)
        proj_proto = model.proto_proj_mlp(true_proto)
        proj_proto = F.normalize(proj_proto, p=2, dim=-1)

        # 將形狀從 (2, 5, C) 攤平為 (10, C)
        true_flat = true_proto.view(-1, true_proto.shape[-1])
        proj_flat = proj_proto.view(-1, proj_proto.shape[-1])

        # 計算 Cosine 相似度矩陣 (10x10)
        sim_true = torch.mm(true_flat, true_flat.t()).numpy()
        sim_proj = torch.mm(proj_flat, proj_flat.t()).numpy()

    # --- 建立視覺化標籤 ---
    labels = [f"BG_{i}" for i in range(num_prototype)] + [f"FG_{i}" for i in range(num_prototype)]

    # --- 輸出文字表格至終端機 ---
    pd.options.display.float_format = '{:.3f}'.format
    print("\n================ [True Prototype] 相似度 ================")
    print(pd.DataFrame(sim_true, index=labels, columns=labels))
    print("\n============== [Projected Prototype] 相似度 ==============")
    print(pd.DataFrame(sim_proj, index=labels, columns=labels))
    print("========================================================\n")

    # --- 繪製並排熱力圖 ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 共同的熱力圖設定
    heatmap_kws = dict(
        annot=True, fmt=".2f", cmap="RdBu_r", 
        vmin=-1.0, vmax=1.0, 
        xticklabels=labels, yticklabels=labels
    )

    # 繪製 左圖: True Prototype
    sns.heatmap(sim_true, ax=axes[0], cbar_kws={'label': 'Cosine Similarity'}, **heatmap_kws)
    axes[0].set_title(f'True Prototype Similarity (Node: {target_node.upper()})', fontsize=16, pad=15)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].axhline(y=num_prototype, color='black', linewidth=2)
    axes[0].axvline(x=num_prototype, color='black', linewidth=2)

    # 繪製 右圖: Projected Prototype
    sns.heatmap(sim_proj, ax=axes[1], cbar_kws={'label': 'Cosine Similarity'}, **heatmap_kws)
    axes[1].set_title('Projected Prototype Similarity (D1 Level)', fontsize=16, pad=15)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].axhline(y=num_prototype, color='black', linewidth=2)
    axes[1].axvline(x=num_prototype, color='black', linewidth=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"對比視覺化結果已儲存至: {save_path}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth_path', type=str, default='models/prototype_cascade/prototypev3_fb55_x4/best.pth')
    parser.add_argument('--target_node', type=str, default='x4')
    parser.add_argument('--save_name', type=str, default='v3_prototype_comparison_x4.png')
    opt = parser.parse_args()
    
    check_v3_similarities(
        model_path=opt.pth_path,
        target_node=opt.target_node,
        save_path=opt.save_name
    )