# 檔案: valid_stage3_aspp.py
import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns # 用於繪製熱力圖
from models.PrototypeSegmenter_ASPP import PrototypeSegmenterASPP
from dataloaders.SupervisedPolypDataset import SupervisedPolypDataset
from util.metric import SegmentationMetric

random.seed(42)

def save_visualization(original_image, true_mask, binary_pred, k_class_pred, output_path):
    """
    生成四聯圖：原圖 | 真值 | 二元預測 | K類子預測
    """
    if original_image.dtype != np.uint8:
        original_image = (original_image * 255).clip(0, 255).astype(np.uint8)
    if len(original_image.shape) == 3 and original_image.shape[0] < 5:
        original_image = np.transpose(original_image, (1, 2, 0))

    h, w, _ = original_image.shape
    
    # Resize masks
    def _resize(m):
        if m.shape != (h, w):
            return cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return m
    
    true_mask = _resize(true_mask)
    binary_pred = _resize(binary_pred)
    k_class_pred = _resize(k_class_pred)

    # 1. Ground Truth (Yellow)
    true_colored = np.zeros_like(original_image)
    true_colored[true_mask == 1] = [255, 255, 0]
    
    # 2. Binary Prediction (Yellow)
    binary_colored = np.zeros_like(original_image)
    binary_colored[binary_pred == 1] = [255, 255, 0]
    
    # 3. K-Class Prediction (Tab10 Colormap)
    # 我們希望不同的子類別有明顯不同的顏色
    # 獲取唯一的類別 ID (排除背景類別 0, 如果有的話)
    unique_labels = np.unique(k_class_pred)
    cmap = cm.get_cmap('tab20')
    k_colored = np.zeros_like(original_image)
    
    for label in unique_labels:
        # 假設 K 類輸出是 0-indexed (0~K-1)
        # 我們可以跳過背景類 (通常是後面的幾個 ID，或者由 binary mask 過濾)
        # 這裡為了展示模型的所有決策，我們為每個 ID 上色
        color = np.array(cmap(label % 20)[:3]) * 255
        k_colored[k_class_pred == label] = color.astype(np.uint8)
    
    # Add Titles
    font = cv2.FONT_HERSHEY_SIMPLEX
    def add_title(img, txt):
        img_copy = img.copy() # cv2 putText is in-place
        # 為了字體清晰，加個白底
        img_copy = cv2.copyMakeBorder(img_copy, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.putText(img_copy, txt, (10, 20), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        return img_copy

    img1 = add_title(cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR), "Original")
    img2 = add_title(cv2.cvtColor(true_colored, cv2.COLOR_RGB2BGR), "Ground Truth")
    img3 = add_title(cv2.cvtColor(binary_colored, cv2.COLOR_RGB2BGR), "Binary Pred")
    img4 = add_title(cv2.cvtColor(k_colored, cv2.COLOR_RGB2BGR), "K-Class Prototypes")

    combined = np.concatenate((img1, img2, img3, img4), axis=1)
    cv2.imwrite(output_path, combined)

def analyze_prototypes(model, output_dir, num_fg, num_bg):
    """
    分析模型學習到的原型：計算相似度矩陣並繪製熱力圖。
    """
    print("\n正在分析原型相似度...")
    # 提取原型權重 (K, C)
    prototypes = model.prototypes.data.cpu().numpy()
    
    # 計算餘弦相似度矩陣 (K, K)
    # 先歸一化
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    prototypes_norm = prototypes / (norms + 1e-8)
    similarity_matrix = np.dot(prototypes_norm, prototypes_norm.T)
    
    # 繪製 Heatmap
    plt.figure(figsize=(12, 10))
    
    # 創建標籤
    labels = [f'FG_{i+1}' for i in range(num_fg)] + [f'BG_{i+1}' for i in range(num_bg)]
    
    sns.heatmap(similarity_matrix, 
                annot=True, fmt=".2f", 
                xticklabels=labels, yticklabels=labels,
                cmap="coolwarm", vmin=-1, vmax=1)
    
    plt.title("Learned Prototypes Cosine Similarity")
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "prototypes_similarity_heatmap.png")
    plt.savefig(save_path)
    plt.close()
    print(f"原型相似度熱力圖已儲存至: {save_path}")

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    # --- 1. 準備模型 ---
    model = PrototypeSegmenterASPP(cfg=cfg['model_cfg']).to(device)
    try:
        print(f"正在載入模型權重: {cfg['reload_model_path']}")
        checkpoint = torch.load(cfg['reload_model_path'], map_location=device)
        if 'model_state_dict' in checkpoint:
            # 如果是新的完整 Checkpoint 格式
            model.load_state_dict(checkpoint['model_state_dict'])
            print("[INFO] Loaded model state dict from full checkpoint.")
        else:
            # 如果是舊的只存權重的格式
            model.load_state_dict(checkpoint)
            print("[INFO] Loaded raw model weights.")
    except Exception as e:
        print(f"錯誤: 無法載入模型權重: {e}")
        return
    model.eval()

    # --- 2. 準備數據 ---
    # 驗證時不需要外部原型，所以 prototype_dir 可以是 None
    val_dataset = SupervisedPolypDataset(
        image_dir=cfg['val_image_dir'],
        mask_dir=cfg['val_mask_dir'],
        target_size=(cfg['image_size'], cfg['image_size']),
        return_original=True,
        prototype_dir=None 
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    output_dir = cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # 隨機選擇要保存的圖像索引
    num_images = len(val_dataset)
    indices_to_save = random.sample(range(num_images), k=min(cfg['num_visualizations'], num_images))
    print(f"將保存索引為 {indices_to_save} 的圖像結果。")

    # --- 3. 評估循環 ---
    metric = SegmentationMetric(num_classes=2)
    
    print("開始評估...")
    with torch.no_grad():
        # 注意：這裡解包 5 個值 (image, mask, prototype_placeholder, original, path)
        # 因為 SupervisedPolypDataset 在 return_original=True 時返回 5 個
        for i, (images, masks, _, original_images, paths) in enumerate(tqdm(val_loader)):
            images = images.to(device)
            
            # 模型前向傳播 (返回二元 Logits 和 K 類 Logits)
            logits_binary, logits_k_class = model(images)
            
            # 獲取預測結果
            pred_binary = logits_binary.argmax(dim=1).cpu().numpy()
            pred_k_class = logits_k_class.argmax(dim=1).cpu().numpy()
            
            # 更新指標
            metric.update(masks.numpy(), pred_binary)
            
            # 保存可視化
            if i in indices_to_save:
                save_path = os.path.join(output_dir, f"result_{os.path.basename(paths[0])}")
                save_visualization(
                    original_image=original_images[0].numpy(),
                    true_mask=masks[0].numpy(),
                    binary_pred=pred_binary[0],
                    k_class_pred=pred_k_class[0],
                    output_path=save_path
                )

    # --- 4. 輸出結果 ---
    scores = metric.get_scores()
    dice = scores["Class_Dice"][1]
    iou = scores["Mean_IoU"]
    
    print("\n" + "="*40)
    print(f"  Validation Dice: {dice:.4f}")
    print(f"  Validation mIoU: {iou:.4f}")
    print("="*40)

    # --- 5. 原型分析 ---
    analyze_prototypes(model, output_dir, 
                       cfg['model_cfg']['num_fg_prototypes'], 
                       cfg['model_cfg']['num_bg_prototypes'])

if __name__ == '__main__':
    config = {
        # 數據路徑
        'val_image_dir': '/home/U116med/data/polyp/TestDataset/test/images/',
        'val_mask_dir': '/home/U116med/data/polyp/TestDataset/test/masks/',
        'reload_model_path': './final_model_aspp_enlr-4/best_model_aspp.pth',
        
        'image_size': 352,
        'num_visualizations': 10,
        'output_dir': './validation_results_aspp_enlr-4/',
        
        # 模型配置 (必須與訓練時一致)
        'model_cfg': {
            'decoder_out_channels': 256,
            'num_fg_prototypes': 4,
            'num_bg_prototypes': 4,
        }
    }
    
    main(config)