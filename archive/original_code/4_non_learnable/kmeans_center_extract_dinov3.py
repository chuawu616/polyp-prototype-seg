import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

# 假設你的模型定義在 lib.networks_dinov3 中
from lib.networks_dinov3_concat import DINOv3_concat
from utils.dataloader import get_loader

def extract_and_cluster_dinov3_features(model, dataloader, num_classes=2, num_prototype=5, max_per_batch=2000):
    model.eval()
    
    features_dict = {i: [] for i in range(num_classes)}
    
    feat_norm = torch.nn.LayerNorm(256).cuda() 
    
    print("開始提取 DINOv3 對齊後的投影特徵...")
    with torch.no_grad():
        for pack in tqdm(dataloader):
            images, gts = pack
            images = images.cuda()
            gts = gts.cuda()
            
            # 1. 處理輸入頻道
            if images.size()[1] == 1:
                images = images.repeat(1, 3, 1, 1)
            
            # 2. 提取 DINOv3 中間層特徵並拼接
            # 參考 DINOv3_concat.forward 邏輯
            feats = model.backbone.get_intermediate_layers(images, n=model.layer_indices, reshape=True, norm=True)
            feat_concat = torch.cat(feats, dim=1)
            
            # 3. 通過 Projection 層 (對應 Prototype 模型中的預處理空間)
            c = model.proj(feat_concat) # (B, 256, H/16, W/16)
            
            b, ch, h, w = c.shape
            
            # 4. 特徵空間對齊：LayerNorm + L2 Normalize
            # 將 (B, C, H, W) -> (B*H*W, C)
            _c = c.permute(0, 2, 3, 1).reshape(-1, ch)
            _c = feat_norm(_c)
            feat = F.normalize(_c, p=2, dim=-1)
            
            # 5. GT 降採樣以匹配 1/16 的特徵圖
            gts_down = F.interpolate(gts, size=(h, w), mode='nearest')
            gts_down = gts_down.view(-1).long()
            
            # 6. 類別分離與隨機抽樣
            for k in range(num_classes):
                mask = (gts_down == k)
                if mask.sum() > 0:
                    class_feat = feat[mask]
                    
                    if class_feat.size(0) > max_per_batch:
                        perm = torch.randperm(class_feat.size(0), device=class_feat.device)
                        idx = perm[:max_per_batch]
                        class_feat = class_feat[idx]
                        
                    features_dict[k].append(class_feat.cpu().numpy())
                    
    # 初始化 Prototype 張量 (num_classes, num_prototype, 256)
    prototype_centers = torch.zeros(num_classes, num_prototype, 256)
    
    print("開始進行 K-Means 分群...")
    for k in range(num_classes):
        if len(features_dict[k]) == 0:
            continue
            
        class_features = np.concatenate(features_dict[k], axis=0)
        print(f"類別 {k} 樣本數: {class_features.shape[0]}")
        
        # MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=num_prototype, batch_size=50000, n_init=3, max_iter=100)
        kmeans.fit(class_features)
        
        # 取得中心並 L2 Normalize
        centers = torch.from_numpy(kmeans.cluster_centers_)
        centers = F.normalize(centers, p=2, dim=-1)
        
        prototype_centers[k] = centers

    return prototype_centers

if __name__ == '__main__':
    # 設定路徑
    train_path = '/home/U116med/data/polyp/TrainDataset/'
    image_root = f'{train_path}/images/'
    gt_root = f'{train_path}/masks/'
    
    # 1. 載入模型
    model = DINOv3_concat(backbone_type='vits16plus', freeze=True).cuda()
    
    pretrained_path = '/home/U116med/wch_code/non_learnable/models/polyp/vits16plus/baseline_dinov3/best.pth'
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path))
        print(f"成功載入預訓練權重: {pretrained_path}")
    
    # 2. 準備數據
    train_loader = get_loader(image_root, gt_root, batchsize=8, trainsize=352)
    
    # 3. 提取中心
    prototypes = extract_and_cluster_dinov3_features(
        model, 
        train_loader, 
        num_classes=2, 
        num_prototype=5
    )
    
    # 4. 儲存
    save_path = 'kmeans_dinov3_prototypes100.pth'
    torch.save(prototypes, save_path)
    print(f"DINOv3 Prototype 已儲存至 {save_path}, Shape: {prototypes.shape}")