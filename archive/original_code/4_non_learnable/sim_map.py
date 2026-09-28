import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

def check_prototype_similarity(weight_path, num_classes=2, num_prototype=5, save_path='proto_sim_table.png'):
    if not os.path.exists(weight_path):
        print(f"找不到權重檔案: {weight_path}")
        return

    print(f"正在讀取權重: {weight_path}")
    # map_location='cpu' 確保即使沒有 GPU 也能跑
    state_dict = torch.load(weight_path, map_location='cpu')
    
    # 直接從字典中把 prototypes 參數抽出來
    # 如果你的模型被包在 DataParallel 裡，key 可能會是 'module.prototypes'
    proto_key = 'prototypes' if 'prototypes' in state_dict else 'module.prototypes'
    
    if proto_key not in state_dict:
        print("在權重檔中找不到 prototypes 參數！請確認模型架構。")
        return
        
    prototypes = state_dict[proto_key] # shape: (2, 5, 64)
    print(f"成功提取 Prototypes，形狀為: {prototypes.shape}")
    
    # 1. 將 (2, 5, 64) 攤平為 (10, 64)
    # 前 5 個是 BG，後 5 個是 FG
    prototypes_flat = prototypes.view(-1, prototypes.shape[-1])
    
    # 2. 進行 L2 正規化 (模擬模型前向傳播時的狀態)
    protos_norm = F.normalize(prototypes_flat, p=2, dim=-1)
    
    # 3. 計算 Cosine 相似度矩陣 (單位向量的內積即為 Cosine 相似度)
    # shape: (10, 64) @ (64, 10) -> (10, 10)
    sim_matrix = torch.mm(protos_norm, protos_norm.t()).numpy()
    
    # --- 建立標籤 ---
    labels = [f"BG_{i}" for i in range(num_prototype)] + [f"FG_{i}" for i in range(num_prototype)]
    
    # --- 輸出為終端機表格 (Pandas) ---
    df = pd.DataFrame(sim_matrix, index=labels, columns=labels)
    print("\n================== Prototype Cosine 相似度表格 ==================")
    # 設定 pandas 顯示格式，只留小數點後 3 位
    pd.options.display.float_format = '{:.3f}'.format
    print(df)
    print("=================================================================\n")
    
    # --- 繪製並儲存視覺化 Heatmap ---
    plt.figure(figsize=(10, 8))
    
    # 使用 seaborn 畫 heatmap，數值標記上去
    sns.heatmap(sim_matrix, 
                annot=True,          # 顯示數字
                fmt=".2f",           # 小數點後兩位
                cmap="RdBu_r",       # 顏色表 (紅-藍，數值越大越偏紅)
                vmin=-1.0, vmax=1.0, # Cosine 相似度範圍 [-1, 1]
                xticklabels=labels, 
                yticklabels=labels,
                cbar_kws={'label': 'Cosine Similarity'})
    
    plt.title('Prototype Cosine Similarity Matrix', fontsize=16, pad=15)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # 畫兩條分隔線，把 BG 和 FG 切開來，方便觀察
    plt.axhline(y=num_prototype, color='black', linewidth=2)
    plt.axvline(x=num_prototype, color='black', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"相似度矩陣熱力圖已儲存至: {save_path}")
    plt.close()

if __name__ == '__main__':
    # 請替換成你訓練完發現分數下降的模型權重路徑
    weight_file = 'models/prototype_cascade/prototype_fb55_pretrained100_add_aug/best.pth' 
    
    check_prototype_similarity(
        weight_path=weight_file, 
        num_classes=2, 
        num_prototype=5,
        save_path='prototype_fb55_pretrained100_add_aug_similarity_table.png'
    )