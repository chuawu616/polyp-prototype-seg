import os
import glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from sklearn.cluster import KMeans
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
from skimage import color # 導入 skimage.color 用於上色
from models.pvtv2 import pvt_v2_b2

class SublabelGenDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(352, 352)):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        self.image_paths.extend(sorted(glob.glob(os.path.join(image_dir, '*.jpg'))))
        self.mask_map = {os.path.splitext(os.path.basename(p))[0]: p
                       for p in glob.glob(os.path.join(mask_dir, '*.png'))}
        self.image_paths = [p for p in self.image_paths if os.path.splitext(os.path.basename(p))[0] in self.mask_map]
        if not self.image_paths:
            raise FileNotFoundError(f"在 {image_dir} 中找不到任何匹配的 影像-遮罩 對。")
        print(f"找到 {len(self.image_paths)} 組影像-遮罩對用於生成子標籤。")
        self.transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = self.mask_map[base_name]
        img_pil = Image.open(img_path).convert("RGB")
        mask_pil = Image.open(mask_path).convert("L")
        image_tensor = self.transform(img_pil)
        mask_np = (np.array(mask_pil) > 128).astype(np.uint8)
        return image_tensor, mask_np, img_path

def save_sublabel_visualization(output_dir, base_name, original_image, binary_mask, sublabel_map):
    """
    儲存用於檢查子標籤生成結果的視覺化圖像。
    """
    if original_image.dtype != np.uint8:
        original_image = np.array(original_image)

    # 為二元遮罩上色 
    true_colored = np.zeros_like(original_image)
    true_colored[binary_mask == 1] = [255, 255, 255] 

    # 為多類別子標籤上色
    # 使用 color.label2rgb 將每個標籤ID映射到不同顏色，並疊加在原圖上
    sublabel_vis = color.label2rgb(sublabel_map, image=original_image, bg_label=0, image_alpha=0.5)
    sublabel_vis = (sublabel_vis * 255).astype(np.uint8)

    # 添加標題
    font = cv2.FONT_HERSHEY_SIMPLEX
    def add_title(image, text):
        img_with_title = cv2.copyMakeBorder(image, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.putText(img_with_title, text, (10, 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        return img_with_title

    img1 = add_title(cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR), "Original")
    img2 = add_title(cv2.cvtColor(true_colored, cv2.COLOR_RGB2BGR), "Binary Mask")
    img3 = add_title(cv2.cvtColor(sublabel_vis, cv2.COLOR_RGB2BGR), "Generated Sublabels")

    combined_image = np.concatenate((img1, img2, img3), axis=1)
    
    output_path = os.path.join(output_dir, f"{base_name}_sublabel_check.png")
    cv2.imwrite(output_path, combined_image)

def generate_sublabels(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    # --- 準備模型和資料 ---
    model = pvt_v2_b2(pretrained=False).to(device).eval()
    try:
        state_dict = torch.load(cfg['stage1_pretrained_path'], map_location=device)
        model.load_state_dict(state_dict)
    except Exception:
        model = pvt_v2_b2(pretrained=True).to(device).eval()
        
    dataset = SublabelGenDataset(
        image_dir=cfg['image_dir'],
        mask_dir=cfg['mask_dir'],
        image_size=cfg['image_size']
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=cfg['num_workers'])
    
    os.makedirs(cfg['output_dir'], exist_ok=True)
    vis_dir = cfg.get('visualization_dir')
    if vis_dir: os.makedirs(vis_dir, exist_ok=True)

    # --- 2. 遍歷資料集生成子標籤 ---
    with torch.no_grad():
        for i, (image, mask_np, path) in enumerate(tqdm(dataloader, desc="生成子標籤")):
            
            base_name = os.path.splitext(os.path.basename(path[0]))[0]
            output_path = os.path.join(cfg['output_dir'], f"{base_name}.png")
            
            if os.path.exists(output_path) and not cfg.get('overwrite', False):
                continue
            
            image = image.to(device)
            true_mask_orig = mask_np[0].numpy()
            
            features = model.forward_features(image)[cfg['target_stage_idx']]
                        
            feat_map_np = np.transpose(features.squeeze(0).cpu().numpy(), (1, 2, 0)) # (h, w, C)
            
            mask_orig_pil = Image.fromarray(true_mask_orig)
            mask_small_pil = mask_orig_pil.resize((feat_map_np.shape[1], feat_map_np.shape[0]), Image.NEAREST)
            mask_small_np = np.array(mask_small_pil) # (h, w)

            pixels_flat = feat_map_np.reshape(-1, feat_map_np.shape[-1])
            mask_flat = mask_small_np.flatten()

            fg_pixels = pixels_flat[mask_flat == 1]
            bg_pixels = pixels_flat[mask_flat == 0]
            
            # 1. 創建一個【整數類型】的 NumPy 數組
            sublabel_map_small = np.zeros_like(mask_small_np, dtype=np.uint8)
            
            # 2. 對前景像素做 K-Means
            if len(fg_pixels) >= cfg['n_fg_clusters']:
                kmeans_fg = KMeans(n_clusters=cfg['n_fg_clusters'], random_state=42, n_init=10)
                fg_labels = kmeans_fg.fit_predict(fg_pixels) # fg_labels 是 int 數組
                # NumPy 的布爾索引賦值，類型兼容性更好
                sublabel_map_small[mask_small_np == 1] = fg_labels + 1
            
            # 3. 對背景像素做 K-Means
            if len(bg_pixels) >= cfg['n_bg_clusters']:
                kmeans_bg = KMeans(n_clusters=cfg['n_bg_clusters'], random_state=42, n_init=10)
                bg_labels = kmeans_bg.fit_predict(bg_pixels)
                sublabel_map_small[mask_small_np == 0] = bg_labels + 1 + cfg['n_fg_clusters']

            # 4. 上采樣並保存 (與之前相同)
            sublabel_map_resized = cv2.resize(sublabel_map_small, 
                                              (cfg['image_size'], cfg['image_size']), 
                                              interpolation=cv2.INTER_NEAREST)
            Image.fromarray(sublabel_map_resized).save(output_path)
            
            # 5. 視覺化
            if vis_dir and i < 10:
                original_image = Image.open(path[0]).convert("RGB").resize((cfg['image_size'], cfg['image_size']))
                original_mask_resized = cv2.resize(true_mask_orig, (cfg['image_size'], cfg['image_size']), interpolation=cv2.INTER_NEAREST)
                save_sublabel_visualization(vis_dir, base_name, np.array(original_image), original_mask_resized, sublabel_map_resized)

    print("\n所有子標籤已成功生成！")

if __name__ == '__main__':
    config = {
        'image_dir': '/home/U116med/data/polyp/TrainDataset/images/',
        'mask_dir': '/home/U116med/data/polyp/TrainDataset/masks/',
        'stage1_pretrained_path': './pretrained_stage1_finetuned_on_test/stage1_epoch_20.pth',
        
        'image_size': 352,
        'batch_size': 1,
        'num_workers': 4,

        'overwrite': False,
        
        'target_stage_idx': 1,
        'n_fg_clusters': 4,
        'n_bg_clusters': 4,
        
        'output_dir': './polypdata/sublabels_k8/',
        'visualization_dir': './sublabel_generation_check/'
    }
            
    generate_sublabels(config)