import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

from lib.networks_pvtv2_cascade import PVT_CASCADE
from lib.networks_prototype_v3 import Prototype_CASCADE_v3
from lib.networks_prototype_v0 import Prototype_CASCADE_v0
from utils.dataloader import test_dataset

def extract_all_node_features(model, images):
    """Extracts features from all potential target nodes in the Backbone and Decoder."""
    if images.size()[1] == 1:
        images = model.conv(images)
        
    with torch.no_grad():
        x1, x2, x3, x4 = model.backbone(images)
        dd4, dd3, dd2, dd1, d1 = model.decoder(x4, [x3, x2, x1])
        
    return {
        'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4,
        'dd4': dd4, 'dd3': dd3, 'dd2': dd2, 'dd1': dd1, 'd1': d1
    }

def apply_pca_to_feature(feat_tensor, target_size=(352, 352)):
    """Reduces the feature map to 3 dimensions via PCA and converts to RGB visualization."""
    feat = feat_tensor.squeeze(0) # (C, H, W)
    C, H, W = feat.shape
    
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
    feat_rgb_tensor = torch.from_numpy(feat_rgb).permute(2, 0, 1).unsqueeze(0).float()
    feat_rgb_up = F.interpolate(feat_rgb_tensor, size=target_size, mode='bilinear', align_corners=False)
    
    return feat_rgb_up.squeeze(0).permute(1, 2, 0).numpy()

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
    # Convert BGR format for cv2 compatibility later
    img_vis_cv2 = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR) 
    
    gt_vis = np.array(gt.resize((img_size, img_size)))
    gt_vis = np.where(gt_vis > 128, 1, 0).astype(np.uint8) # 0 and 1 for the visualization function
    
    return img_tensor, img_vis_cv2, gt_vis

def get_prototype_worst_case_img(original_image, true_mask, binary_pred, k_class_pred, dice_score, fg_num, bg_num):
    """
    Generates a four-panel image: Original | Ground Truth | Binary Pred (with score) | K-Class Sub-predictions
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

    # [A] Original
    vis_original = add_title_and_border(original_image.copy(), "Original")

    # [B] Ground Truth (Yellow = FG)
    vis_gt = original_image.copy()
    overlay = vis_gt.copy()
    overlay[true_mask == 1] = [0, 255, 255]
    vis_gt = cv2.addWeighted(overlay, 0.5, vis_gt, 0.5, 0)
    vis_gt = add_title_and_border(vis_gt, "Ground Truth")

    # [C] Binary Prediction (Green = Pred) + Display DICE score
    vis_binary = original_image.copy()
    overlay = vis_binary.copy()
    overlay[binary_pred == 1] = [0, 255, 0]
    vis_binary = cv2.addWeighted(overlay, 0.5, vis_binary, 0.5, 0)
    title_text = f"Binary (DICE: {dice_score:.4f})"
    vis_binary = add_title_and_border(vis_binary, title_text, text_color=(0, 0, 255))
    
    # [D] K-Class Prototypes
    bg_palette = np.array([
    [0, 255, 0], [255, 100, 255], [0, 255, 255], [255, 255, 0],   
    [255, 150, 0], [255, 50, 50], [100, 200, 255], [255, 255, 255], 
    [150, 255, 0], [0, 255, 150], [200, 100, 255], [255, 0, 150],   
    [100, 255, 200], [255, 200, 100], [50, 150, 255], [200, 255, 100], 
    [180, 180, 255], [255, 180, 180], [180, 255, 180], [220, 220, 220]  
    ], dtype=np.uint8)

    fg_palette = np.array([
    [0, 0, 80], [50, 0, 0], [0, 50, 0], [50, 50, 0],                
    [50, 0, 50], [0, 50, 50], [30, 30, 30], [0, 0, 0],              
    [0, 0, 50], [30, 0, 0], [0, 30, 0], [40, 40, 40],                
    [20, 20, 60], [60, 20, 20], [20, 60, 20], [60, 60, 0],           
    [40, 0, 80], [80, 0, 40], [0, 80, 40], [20, 20, 20]             
    ], dtype=np.uint8)

    k_colored_map = np.zeros((h, w, 3), dtype=np.uint8)    
    for fg_id in range(min(len(fg_palette), fg_num)):
        k_colored_map[k_class_pred == fg_id] = fg_palette[fg_id]
    for bg_id in range(min(len(bg_palette), bg_num)):
        k_colored_map[k_class_pred == (fg_num + bg_id)] = bg_palette[bg_id]
    vis_k = add_title_and_border(k_colored_map, "K-Class Prototypes")

    # Concatenate horizontally into a four-panel image
    combined = np.concatenate((vis_original, vis_gt, vis_binary, vis_k), axis=1)
    # Convert BGR back to RGB for matplotlib
    combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
    return combined_rgb

# ==========================================
# Main Execution
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default='models/prototype_cascade/prototype_v0_fb33/best.pth')
    parser.add_argument('--dataset', type=str, default='test', help='Dataset to analyze')
    parser.add_argument('--worst_k', type=int, default=30, help='Number of worst cases to visualize')
    parser.add_argument('--save_dir', type=str, default='./worst_cases_analysis_v0_fb33/')
    parser.add_argument('--target_node', type=str, default='d1', help='The node the prototype was trained on')
    parser.add_argument('--num_prototype', type=int, default=3)
    opt = parser.parse_args()
    
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    
    # model = Prototype_CASCADE_v3(num_classes=2, num_prototype=5, target_node=opt.target_node).cuda()
    model = Prototype_CASCADE_v0(num_classes=2, num_prototype=3).cuda()
    model.load_state_dict(torch.load(opt.pth_path))
    model.eval()
    
    root_path = '/home/U116med/data/polyp/TestDataset/'
    data_path = os.path.join(root_path, opt.dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    
    print(f'Stage 1: Evaluating {data_path} to find the worst {opt.worst_k} cases...')
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    num_images = len(os.listdir(gt_root))
    
    results_list = []
    
    # --- Stage 1: Calculate Dice for all images ---
    with torch.no_grad():
        for i in tqdm(range(num_images)):
            image_tensor, gt_mask, name = test_loader.load_data()
            
            gt_mask = np.asarray(gt_mask, np.float32)
            gt_mask /= (gt_mask.max() + 1e-8)
            image_tensor = image_tensor.cuda()

            logits_binary, logits_k_class = model(image_tensor)
            
            res = (logits_binary[:, 1:2, :, :] - logits_binary[:, 0:1, :, :]) * 10.0
            res = F.interpolate(res, size=gt_mask.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            
            pred_binary = np.where(res >= 0.5, 1, 0)
            gt_binary = np.where(gt_mask >= 0.5, 1, 0)
            
            sim_map_resized = F.interpolate(logits_k_class, size=gt_mask.shape, mode='bilinear', align_corners=False)
            pred_k_mask = sim_map_resized.argmax(dim=1).squeeze(0).cpu().numpy()
            
            smooth = 1
            intersection = (pred_binary.reshape(-1) * gt_binary.reshape(-1)).sum()
            dice = (2 * intersection + smooth) / (pred_binary.sum() + gt_binary.sum() + smooth)
        
            img_full_path = os.path.join(image_root, name)
            gt_full_path = os.path.join(gt_root, name)
        
            results_list.append({
                'name': name,
                'dice': float(dice),
                'img_path': img_full_path,
                'gt_path': gt_full_path
            })
        
    # Sort by ascending Dice score
    results_list.sort(key=lambda x: x['dice'])
    # worst_cases = results_list[:opt.worst_k]
    worst_cases = results_list[-opt.worst_k:]
    
    # record_file = os.path.join(opt.save_dir, f'worst_{opt.worst_k}_{opt.dataset}_record.txt')
    record_file = os.path.join(opt.save_dir, f'best_{opt.worst_k}_{opt.dataset}_record.txt')
    
    with open(record_file, 'w') as f:
        f.write(f"Worst {opt.worst_k} cases for {opt.dataset}\n")
        f.write("-" * 50 + "\n")
        for rank, item in enumerate(worst_cases):
            line = f"Rank {rank+1:02d} | DICE: {item['dice']:.4f} | File: {item['name']}\n"
            print(line.strip())
            f.write(line)
            
    # --- Stage 2: Retrieve features, predictions, and generate visualization ---
    print(f'\nStage 2: Generating Combined PCA & Prediction visualizations for the worst {opt.worst_k} cases...')
    
    nodes_to_visualize = ['dd1', 'dd2', 'dd3', 'dd4', 'd1', 'x1', 'x2', 'x3']
    # 1 column for the 4-panel image, plus columns for each PCA node
    num_cols = 1 + len(nodes_to_visualize) 
    
    # Calculate appropriate widths. The 4-panel image is 4x wider than a standard PCA image.
    width_ratios = [4] + [1] * len(nodes_to_visualize)
    fig, axes = plt.subplots(opt.worst_k, num_cols, 
                             figsize=(4 * num_cols + 12, 4 * opt.worst_k), # Adjust total figure width
                             gridspec_kw={'width_ratios': width_ratios})
    
    for row_idx, item in enumerate(tqdm(worst_cases)):
        img_tensor, img_vis_cv2, gt_vis = load_image_and_gt_tensor(item['img_path'], item['gt_path'])
        
        with torch.no_grad():
            # Get segmentation and similarity map (for K-class)
            logits_high, similarity_map_high = model(img_tensor)
            
            res_high = (logits_high[:, 1:2, :, :] - logits_high[:, 0:1, :, :]) * 10.0
            pred = res_high.sigmoid().data.cpu().numpy().squeeze()
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            binary_pred = np.where(pred >= 0.5, 1, 0)
            
            # Process K-Class Prediction from the similarity map
            # similarity_map_high shape: (1, 10, H, W) -> Extract max indices
            sim_map_sq = similarity_map_high.squeeze(0) # (10, H, W)
            k_class_pred = torch.argmax(sim_map_sq, dim=0).cpu().numpy() # (H, W), values 0-9
            
            # Get features for PCA
            node_features = extract_all_node_features(model, img_tensor)
        
        # 1. Generate the Four-Panel Image
        four_panel_img = get_prototype_worst_case_img(
            original_image=img_vis_cv2, 
            true_mask=gt_vis, 
            binary_pred=binary_pred, 
            k_class_pred=k_class_pred, 
            dice_score=item['dice'], 
            fg_num=opt.num_prototype, 
            bg_num=opt.num_prototype
        )
        
        # Draw the Four-Panel Image in the first column
        ax_main = axes[row_idx, 0]
        ax_main.imshow(four_panel_img)
        # Add Rank and Name above the main block
        ax_main.set_title(f"Rank {row_idx+1:02d} | File: {item['name']}", fontsize=14, pad=10, loc='left')
        ax_main.axis('off')
        
        # 2. Generate and draw PCA for each requested node
        for col_idx, node in enumerate(nodes_to_visualize):
            feat_rgb_vis = apply_pca_to_feature(node_features[node])
            ax_pca = axes[row_idx, 1 + col_idx]
            
            # Highlight the target_node to easily identify it in the lineup
            title_color = 'red' if node == opt.target_node else 'black'
            ax_pca.imshow(feat_rgb_vis)
            ax_pca.set_title(f"PCA: {node.upper()}", fontsize=14, color=title_color)
            ax_pca.axis('off')

    plt.tight_layout()
    # save_fig_path = os.path.join(opt.save_dir, f'worst_{opt.worst_k}_{opt.dataset}_v3_analysis.jpg')
    save_fig_path = os.path.join(opt.save_dir, f'best_{opt.worst_k}_{opt.dataset}_v0_analysis.jpg')
   
    # Save with reasonable DPI to manage file size given the very large dimensions
    plt.savefig(save_fig_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    print(f"\nExecution Complete!")
    print(f"Record saved to: {record_file}")
    print(f"Visualization saved to: {save_fig_path}")