import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import argparse

# 假設 MetricSegmenter 可被導入
from models.MetricSegmenter import MetricSegmenter

def analyze_prototypes(model_path, model_cfg):
    """
    加載模型，提取原型，並進行分析。
    """
    print(f"--- 正在分析模型: {model_path} ---")
    
    # --- 1. 加載模型和原型 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 實例化模型結構
    model = MetricSegmenter(cfg=model_cfg).to(device)
    # 加載訓練好的權重
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 提取可學習的前景原型
    # .data 將其從計算圖中分離出來，.cpu().numpy() 轉換為 NumPy
    prototypes = model.fg_prototypes.data.cpu().numpy()
    
    num_prototypes, feature_dim = prototypes.shape
    print(f"成功提取 {num_prototypes} 個前景原型，每個維度為 {feature_dim}。")

    # --- 2. 定量分析：計算原型間的相似度 ---
    print("\n--- 定量分析 (相似度矩陣) ---")
    
    # 計算余弦相似度矩陣
    sim_matrix = cosine_similarity(prototypes)
    
    print("前景原型間的餘弦相似度矩陣:")
    # 使用格式化輸出，使其更易讀
    header = [f"P_{i}" for i in range(num_prototypes)]
    print("{:>5}".format(""), end="")
    for h in header: print("{:>7}".format(h), end="")
    print()
    for i, row in enumerate(sim_matrix):
        print("{:>5}".format(f"P_{i}"), end="")
        for val in row:
            print(f"{val:7.3f}", end="")
        print()
        
    # 提取非對角線元素以計算統計數據
    off_diagonal_indices = np.triu_indices(num_prototypes, k=1)
    if len(off_diagonal_indices[0]) > 0:
        pairwise_similarities = sim_matrix[off_diagonal_indices]
        
        print("\n統計摘要:")
        print(f"  - 平均相似度: {pairwise_similarities.mean():.4f}")
        print(f"  - 相似度標準差: {pairwise_similarities.std():.4f}")
        print(f"  - 最高相似度 (非自身): {pairwise_similarities.max():.4f}")
        print(f"  - 最低相似度: {pairwise_similarities.min():.4f}")

    # --- 3. 定性分析：視覺化相似度矩陣 ---
    print("\n--- 定性分析 (熱圖視覺化) ---")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(sim_matrix, annot=True, cmap='viridis', fmt='.2f', 
                xticklabels=header, yticklabels=header)
    plt.title(f"Cosine Similarity of {num_prototypes} Foreground Prototypes")
    plt.tight_layout()
    
    # 保存熱圖
    output_filename = f"prototype_similarity_heatmap_{os.path.basename(model_path).replace('.pth','')}.png"
    plt.savefig(output_filename)
    print(f"相似度熱圖已保存至: {output_filename}")
    # plt.show() # 如果在本地運行，可以取消註釋以直接顯示圖像

def main():
    parser = argparse.ArgumentParser(description="分析 MetricSegmenter 模型的前景原型以檢查聚類坍塌。")
    parser.add_argument("model_path", type=str, help="要分析的 .pth 模型權重檔案的路徑。")
    
    # 添加模型配置參數，因為實例化模型需要它們
    # 這些值應與訓練時使用的值匹配
    parser.add_argument("--num_fg_prototypes", type=int, default=4, help="前景原型的數量。")
    parser.add_argument("--decoder_out_channels", type=int, default=256, help="FPN 解碼器的輸出通道數。")
    
    args = parser.parse_args()
    
    # 構建模型配置文件
    model_config = {
        'num_fg_prototypes': args.num_fg_prototypes,
        'decoder_out_channels': args.decoder_out_channels,
    }
    
    analyze_prototypes(args.model_path, model_config)

if __name__ == '__main__':
    main()