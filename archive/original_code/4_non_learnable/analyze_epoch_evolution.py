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

from lib.networks_prototype_v3 import Prototype_CASCADE_v3

def extract_all_node_features(model, images):
    """Extracts features from all potential target nodes in the Backbone and Decoder."""
    if images.size()[1] == 1:
        images = model.conv(images)
        
    with torch.no_grad():
        x1, x2, x3, x4 = model.backbone(images)
        dd4, dd3, dd2, dd1, d1 = model.decoder(x4, [x3, x2, x1])
        
    return {
        'x1': x1, 'x2': x2, 'x3': x3,
        'dd4': dd4, 'dd3': dd3, 'dd2': dd2, 'dd1': dd1, 'd1': d1
    }

def apply_pca_to_feature(feat_tensor, target_size=(352, 352)):
    """Upsamples the feature map FIRST, then reduces to 3 dimensions via PCA for high-res visualization."""
    
    feat_up = F.interpolate(feat_tensor, size=target_size, mode='bilinear', align_corners=False)
    
    feat = feat_up.squeeze(0) # (C, H, W)，此時 H, W 已是 352, 352
    C, H, W = feat.shape
    
    # 此時 N_samples = 352 * 352 = 123904
    feat_flat = feat.permute(1, 2, 0).reshape(-1, C).cpu().numpy()
    
    pca = PCA(n_components=3)
    feat_pca = pca.fit_transform(feat_flat)
    
    feat_pca_norm = np.zeros_like(feat_pca)
    for i in range(3):
        min_val = feat_pca[:, i].min()
        max_val = feat_pca[:, i].max()
        if max_val > min_val:
            feat_pca_norm[:, i] = (feat_pca[:, i] - min_val) / (max_val - min_val)
            
    feat_rgb = feat_pca_norm.reshape(H, W, 3)
    
    return feat_rgb

def load_image_and_gt_tensor(img_path, gt_path, img_size=352):
    image = Image.open(img_path).convert('RGB')
    gt = Image.open(gt_path).convert('L')
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).cuda()
    
    img_vis = np.array(image.resize((img_size, img_size)))
    img_vis_cv2 = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR) 
    
    gt_vis = np.array(gt.resize((img_size, img_size)))
    gt_vis = np.where(gt_vis > 128, 1, 0).astype(np.uint8) 
    
    return img_tensor, img_vis_cv2, gt_vis

def get_prototype_evolution_img(original_image, true_mask, binary_pred, k_class_pred, epoch, fg_num, bg_num):
    """
    Generates a four-panel image for a specific Epoch
    """
    h, w, _ = original_image.shape
    
    def _resize(m):
        if m.shape != (h, w):
            return cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return m
    
    true_mask = _resize(true_mask)
    binary_pred = _resize(binary_pred)
    k_class_pred = _resize(k_class_pred)

    font = cv2.FONT_HERSHEY_SIMPLEX
    def add_title_and_border(img, txt, text_color=(0, 0, 0)):
        img = cv2.copyMakeBorder(img, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.putText(img, txt, (10, 30), font, 0.7, text_color, 2, cv2.LINE_AA)
        return img

    vis_original = add_title_and_border(original_image.copy(), f"Epoch {epoch} | Original")

    vis_gt = original_image.copy()
    overlay = vis_gt.copy()
    overlay[true_mask == 1] = [0, 255, 255]
    vis_gt = cv2.addWeighted(overlay, 0.5, vis_gt, 0.5, 0)
    vis_gt = add_title_and_border(vis_gt, "Ground Truth")

    vis_binary = original_image.copy()
    overlay = vis_binary.copy()
    overlay[binary_pred == 1] = [0, 255, 0]
    vis_binary = cv2.addWeighted(overlay, 0.5, vis_binary, 0.5, 0)
    vis_binary = add_title_and_border(vis_binary, "Binary Pred", text_color=(0, 0, 255))
    
    fg_palette = np.array([
    [0, 255, 0], [255, 100, 255], [0, 255, 255], [255, 255, 0],   
    [255, 150, 0], [255, 50, 50], [100, 200, 255], [255, 255, 255], 
    [150, 255, 0], [0, 255, 150], [200, 100, 255], [255, 0, 150],   
    [100, 255, 200], [255, 200, 100], [50, 150, 255], [200, 255, 100], 
    [180, 180, 255], [255, 180, 180], [180, 255, 180], [220, 220, 220]
    ], dtype=np.uint8)

    bg_palette = np.array([
    [0, 0, 80], [50, 0, 0], [0, 50, 0], [50, 50, 0],                
    [50, 0, 50], [0, 50, 50], [30, 30, 30], [0, 0, 0],              
    [0, 0, 50], [30, 0, 0], [0, 30, 0], [40, 40, 40],                
    [20, 20, 60], [60, 20, 20], [20, 60, 20], [60, 60, 0],           
    [40, 0, 80], [80, 0, 40], [0, 80, 40], [20, 20, 20]               
    ], dtype=np.uint8)

    k_colored_map = np.zeros((h, w, 3), dtype=np.uint8)    
    for fg_id in range(min(len(fg_palette), fg_num)):
        k_colored_map[k_class_pred == (bg_num + fg_id)] = fg_palette[fg_id] # 修復 K-class 顏色對應邏輯
    for bg_id in range(min(len(bg_palette), bg_num)):
        k_colored_map[k_class_pred == bg_id] = bg_palette[bg_id]
        
    vis_k = add_title_and_border(k_colored_map, "K-Class Prototypes")

    combined = np.concatenate((vis_original, vis_gt, vis_binary, vis_k), axis=1)
    combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
    return combined_rgb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=352)
    parser.add_argument('--model_dir', type=str, default='models/prototype_cascade/prototypev3_fb55_x4')
    parser.add_argument('--img_path', type=str, default="/home/U116med/data/polyp/TrainDataset/images/20.png")
    parser.add_argument('--gt_path', type=str, default="/home/U116med/data/polyp/TrainDataset/masks/20.png")
    parser.add_argument('--save_dir', type=str, default='./collapse_analysis_x4/')
    parser.add_argument('--target_node', type=str, default='x4')
    parser.add_argument('--epochs', type=int, nargs='+', default=[1, 2, 3, 4, 5, 10, 20], help='Epochs to visualize')
    opt = parser.parse_args()
    
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
        
    img_name = os.path.basename(opt.img_path).split('.')[0]
    
    nodes_to_visualize = ['dd2', 'dd3', 'dd4', 'd1', 'x1', 'x2', 'x3']
    num_cols = 1 + len(nodes_to_visualize) 
    width_ratios = [4] + [1] * len(nodes_to_visualize)
    
    fig, axes = plt.subplots(len(opt.epochs), num_cols, 
                             figsize=(4 * num_cols + 12, 4 * len(opt.epochs)),
                             gridspec_kw={'width_ratios': width_ratios})
    
    # 若只有一個 Epoch，強制將 axes 轉為 2D 陣列以利索引
    if len(opt.epochs) == 1:
        axes = np.expand_dims(axes, axis=0)
        
    img_tensor, img_vis_cv2, gt_vis = load_image_and_gt_tensor(opt.img_path, opt.gt_path, opt.img_size)
    
    print(f"Starting Evolution Analysis for Image: {img_name}")
    
    for row_idx, epoch in enumerate(tqdm(opt.epochs)):
        weight_path = os.path.join(opt.model_dir, f'epoch_{epoch}.pth')
        if not os.path.exists(weight_path):
            print(f"Warning: {weight_path} not found. Skipping.")
            continue
            
        model = Prototype_CASCADE_v3(num_classes=2, num_prototype=5, target_node=opt.target_node).cuda()
        model.load_state_dict(torch.load(weight_path, map_location='cuda'))
        model.eval()
        
        with torch.no_grad():
            logits_high, similarity_map_high = model(img_tensor)
            
            res_high = (logits_high[:, 1:2, :, :] - logits_high[:, 0:1, :, :]) * 10.0
            pred = res_high.sigmoid().data.cpu().numpy().squeeze()
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            binary_pred = np.where(pred >= 0.5, 1, 0)
            
            sim_map_sq = similarity_map_high.squeeze(0) # (10, H, W)
            k_class_pred = torch.argmax(sim_map_sq, dim=0).cpu().numpy()
            
            node_features = extract_all_node_features(model, img_tensor)
            
        # 1. 繪製 4-Panel 基礎預測圖
        four_panel_img = get_prototype_evolution_img(
            original_image=img_vis_cv2, 
            true_mask=gt_vis, 
            binary_pred=binary_pred, 
            k_class_pred=k_class_pred, 
            epoch=epoch, 
            fg_num=5, 
            bg_num=5
        )
        
        ax_main = axes[row_idx, 0]
        ax_main.imshow(four_panel_img)
        ax_main.axis('off')
        
        # 2. 繪製並排的各個特徵層 PCA 圖
        for col_idx, node in enumerate(nodes_to_visualize):
            feat_rgb_vis = apply_pca_to_feature(node_features[node])
            ax_pca = axes[row_idx, 1 + col_idx]
            
            title_color = 'red' if node == opt.target_node else 'black'
            ax_pca.imshow(feat_rgb_vis)
            ax_pca.set_title(f"PCA: {node.upper()}", fontsize=16, color=title_color)
            ax_pca.axis('off')

    plt.tight_layout()
    save_fig_path = os.path.join(opt.save_dir, f'{img_name}_evolution_analysis.jpg')
    plt.savefig(save_fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    print(f"\nAnalysis Image saved to: {save_fig_path}")