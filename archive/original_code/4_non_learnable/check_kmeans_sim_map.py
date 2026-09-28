import os
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def check_kmeans_similarity(weight_path, num_classes=2, num_prototype=5, save_path='kmeans_sim_table.png'):
    if not os.path.exists(weight_path):
        print(f"找不到檔案: {weight_path}")
        return

    print(f"正在讀取 K-Means 中心: {weight_path}")
    
    # 讀取 Tensor (map_location='cpu' 確保無需 GPU 也能跑)
    prototypes = torch.load(weight_path, map_location='cpu')
    print(f"成功載入，形狀為: {prototypes.shape}")
    
    # 防呆檢查：確認形狀是否符合預期 (num_classes, num_prototype, ch_dim)
    if len(prototypes.shape) != 3 or prototypes.shape[0] != num_classes or prototypes.shape[1] != num_prototype:
        print(f"警告：Tensor 形狀 {prototypes.shape} 與預期的 ({num_classes}, {num_prototype}, C) 不符！")
    
    # 1. 將 (2, 5, 128) 攤平為 (10, 128)
    prototypes_flat = prototypes.view(-1, prototypes.shape[-1])
    
    # 2. 進行 L2 正規化 (雖然儲存前做過了，但再做一次確保內積絕對等於 Cosine 相似度)
    protos_norm = F.normalize(prototypes_flat, p=2, dim=-1)
    
    # 3. 計算 Cosine 相似度矩陣
    sim_matrix = torch.mm(protos_norm, protos_norm.t()).numpy()
    
    # --- 建立標籤 ---
    labels = [f"BG_{i}" for i in range(num_prototype)] + [f"FG_{i}" for i in range(num_prototype)]
    
    # --- 輸出為終端機表格 (Pandas) ---
    df = pd.DataFrame(sim_matrix, index=labels, columns=labels)
    print("\n================ K-Means 中心 Cosine 相似度表格 ================")
    pd.options.display.float_format = '{:.3f}'.format
    print(df)
    print("=================================================================\n")
    
    # --- 繪製並儲存視覺化 Heatmap ---
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(sim_matrix, 
                annot=True,          
                fmt=".2f",           
                cmap="RdBu_r",       
                vmin=-1.0, vmax=1.0, 
                xticklabels=labels, 
                yticklabels=labels,
                cbar_kws={'label': 'Cosine Similarity'})
    
    # 標題自動帶入檔名
    filename = os.path.basename(weight_path)
    plt.title(f'K-Means Prototype Similarity\n({filename})', fontsize=14, pad=15)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # 畫兩條分隔線，切開 BG 和 FG
    plt.axhline(y=num_prototype, color='black', linewidth=2)
    plt.axvline(x=num_prototype, color='black', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"相似度矩陣熱力圖已儲存至: {save_path}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth_path', type=str, default='kmeans_prototypes_x2_ch128.pth', help='K-Means 特徵中心檔案路徑')
    parser.add_argument('--save_name', type=str, default='kmeans_sim_table_x2.png', help='輸出的圖檔名稱')
    opt = parser.parse_args()
    
    check_kmeans_similarity(
        weight_path=opt.pth_path, 
        num_classes=2, 
        num_prototype=5,
        save_path=opt.save_name
    )