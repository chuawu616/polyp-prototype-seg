import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import argparse

def analyze_single_prototype_file(prototype_path, num_fg_prototypes=8):
    """
    加載原型檔案，計算並打印相似度/距離矩陣。

    Args:
        prototype_path (str): .npy 原型檔案的路徑。
        num_fg_prototypes (int): 前景原型的數量，用於標記矩陣。
    """
    # --- 1. 載入原型檔案 ---
    try:
        prototypes = np.load(prototype_path)
        print(f"成功載入原型檔案: {prototype_path}")
        print(f"原型形狀: {prototypes.shape}")
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 '{prototype_path}'")
        return
    except Exception as e:
        print(f"載入檔案時發生錯誤: {e}")
        return

    # 檢查原型數量是否符合預期
    num_total_prototypes, feature_dim = prototypes.shape
    if num_total_prototypes != 16 or feature_dim != 128:
        print(f"警告: 原型形狀 {prototypes.shape} 與預期的 (16, 128) 不符，但仍會繼續計算。")

    num_bg_prototypes = num_total_prototypes - num_fg_prototypes

    # --- 2. 計算相似度與距離 ---
    print("\n正在計算矩陣...")
    cos_sim_matrix = cosine_similarity(prototypes)
    euclidean_dist_matrix = euclidean_distances(prototypes)

    # --- 3. 格式化並打印結果 ---
    
    # 創建標頭，方便閱讀
    header = [""] + [f"FG_{i}" for i in range(num_fg_prototypes)] + \
             [f"BG_{i}" for i in range(num_bg_prototypes)]
    
    print("\n" + "="*80)
    print("                餘弦相似度 (Cosine Similarity) - 值越高越相似")
    print("="*80)
    
    # 打印帶有標頭的矩陣
    print("{:>6}".format(header[0]), end="")
    for h in header[1:]:
        print("{:>7}".format(h), end="")
    print()
    
    for i, row in enumerate(cos_sim_matrix):
        row_label = header[i+1]
        print("{:>6}".format(row_label), end="")
        for val in row:
            print(f"{val:7.3f}", end="")
        print()
        if i == num_fg_prototypes - 1: # 在前景和背景之間畫一條分隔線
            print("-" * (6 + 7 * num_total_prototypes))


    print("\n" + "="*80)
    print("                歐幾里得距離 (Euclidean Distance) - 值越低越相似")
    print("="*80)

    print("{:>6}".format(header[0]), end="")
    for h in header[1:]:
        print("{:>7}".format(h), end="")
    print()

    for i, row in enumerate(euclidean_dist_matrix):
        row_label = header[i+1]
        print("{:>6}".format(row_label), end="")
        for val in row:
            print(f"{val:7.2f}", end="") # 距離通常數值較大，小數點後兩位即可
        print()
        if i == num_fg_prototypes - 1:
            print("-" * (6 + 7 * num_total_prototypes))

    # --- 4. 計算並打印統計摘要 ---
    print("\n" + "="*80)
    print("                        統計摘要")
    print("="*80)
    
    if num_fg_prototypes > 1:
        fg_vs_fg_sim = cos_sim_matrix[:num_fg_prototypes, :num_fg_prototypes][np.triu_indices(num_fg_prototypes, k=1)].mean()
        fg_vs_fg_dist = euclidean_dist_matrix[:num_fg_prototypes, :num_fg_prototypes][np.triu_indices(num_fg_prototypes, k=1)].mean()
        print(f"前景原型內部平均相似度 (FG vs FG): {fg_vs_fg_sim:.4f}")
        print(f"前景原型內部平均距離 (FG vs FG): {fg_vs_fg_dist:.2f}")
    
    if num_bg_prototypes > 1:
        bg_vs_bg_sim = cos_sim_matrix[num_fg_prototypes:, num_fg_prototypes:][np.triu_indices(num_bg_prototypes, k=1)].mean()
        bg_vs_bg_dist = euclidean_dist_matrix[num_fg_prototypes:, num_fg_prototypes:][np.triu_indices(num_bg_prototypes, k=1)].mean()
        print(f"背景原型內部平均相似度 (BG vs BG): {bg_vs_bg_sim:.4f}")
        print(f"背景原型內部平均距離 (BG vs BG): {bg_vs_bg_dist:.2f}")
        
    fg_vs_bg_sim = cos_sim_matrix[:num_fg_prototypes, num_fg_prototypes:].mean()
    fg_vs_bg_dist = euclidean_dist_matrix[:num_fg_prototypes, num_fg_prototypes:].mean()
    print(f"前景 vs 背景原型平均相似度: {fg_vs_bg_sim:.4f}")
    print(f"前景 vs 背景原型平均距離: {fg_vs_bg_dist:.2f}")
    print("="*80)

def main():
    # 使用 argparse 來讓用戶從命令列傳入檔案路徑
    parser = argparse.ArgumentParser(description="計算並顯示單個原型檔案的內部相似度矩陣。")
    parser.add_argument("prototype_file", type=str, help="要分析的 .npy 原型檔案的路徑。")
    parser.add_argument("--num_fg", type=int, default=8, help="前景原型的數量。")
    
    args = parser.parse_args()
    
    analyze_single_prototype_file(args.prototype_file, args.num_fg)

if __name__ == '__main__':
    # 確保您已安裝 scikit-learn: pip install scikit-learn
    main()