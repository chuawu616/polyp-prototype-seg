import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from lib.networks_pvtv2_cascade import PVT_CASCADE
from utils.dataloader import get_loader

def extract_and_cluster_features(model, dataloader, target_node='dd2', num_classes=2, num_prototype=5, max_per_batch=20000):
    model.eval()
    
    # 用來存放各類別特徵的 list
    features_dict = {i: [] for i in range(num_classes)}
    
    # 用來自動記錄該節點的 Channel 維度
    ch_dim = None 
    
    print(f"開始提取 [{target_node}] 節點對齊後的特徵...")
    with torch.no_grad():
        for pack in tqdm(dataloader):
            images, gts = pack
            images = images.cuda()
            gts = gts.cuda()
            
            # 單通道轉換
            if images.size()[1] == 1:
                images = model.conv(images)
                
            # 1. 通過 Backbone
            x1, x2, x3, x4 = model.backbone(images)
            
            # 2. 通過 Decoder
            dd4, dd3, dd2, dd1, x4_oo = model.decoder(x4, [x3, x2, x1])
            
            # 3. 節點字典映射
            feats = {
                'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4,
                'dd4': dd4, 'dd3': dd3, 'dd2': dd2, 'dd1': dd1, 'd1': x4_oo
            }
            
            if target_node not in feats:
                raise ValueError(f"不支援的節點名稱: {target_node}")
                
            c = feats[target_node]
            b, ch, h, w = c.shape
            
            if ch_dim is None:
                ch_dim = ch
            
            # 攤平
            _c = c.permute(0, 2, 3, 1).reshape(-1, ch) # -> (B*H*W, ch)
            
            # 4. 對齊網路中的 self.feat_norm_target 與 l2_normalize
            # 這裡使用 F.layer_norm 模擬未經訓練的 nn.LayerNorm (gamma=1, beta=0)
            _c = F.layer_norm(_c, _c.shape[-1:])
            feat = F.normalize(_c, p=2, dim=-1) 
            
            # 將 GT 降採樣到與當前特徵圖相同大小
            gts_down = F.interpolate(gts, size=(h, w), mode='nearest')
            gts_down = gts_down.view(-1).long() 
            
            # 根據類別分離特徵，並在 Batch 內進行隨機抽樣避免 OOM
            for k in range(num_classes):
                mask = (gts_down == k)
                if mask.sum() > 0:
                    class_feat = feat[mask]
                    
                    # 限制單一 batch 收集的特徵數量
                    if class_feat.size(0) > max_per_batch:
                        perm = torch.randperm(class_feat.size(0), device=class_feat.device)
                        idx = perm[:max_per_batch]
                        class_feat = class_feat[idx]
                        
                    features_dict[k].append(class_feat.cpu().numpy())
                    
    # 初始化儲存 Prototype 的張量，動態使用偵測到的 ch_dim
    prototype_centers = torch.zeros(num_classes, num_prototype, ch_dim)
    
    print(f"開始進行 K-Means 分群 (特徵維度: {ch_dim})...")
    for k in range(num_classes):
        if len(features_dict[k]) == 0:
            print(f"警告：類別 {k} 沒有提取到任何特徵。")
            continue
            
        # 合併該類別所有批次抽樣後的特徵
        class_features = np.concatenate(features_dict[k], axis=0)
        print(f"類別 {k} 參與分群的特徵數量: {class_features.shape[0]}")
        
        # 使用 MiniBatchKMeans 加速
        kmeans = MiniBatchKMeans(n_clusters=num_prototype, batch_size=50000, n_init=3, max_iter=100)
        kmeans.fit(class_features)
        
        # 取得分群中心並做 L2 正規化
        centers = torch.from_numpy(kmeans.cluster_centers_)
        centers = F.normalize(centers, p=2, dim=-1)
        
        prototype_centers[k] = centers

    return prototype_centers, ch_dim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', type=str, default='x3', choices=['dd1', 'dd2', 'dd3', 'dd4', 'd1', 'x1', 'x2', 'x3'], help='選擇提取特徵的節點')
    parser.add_argument('--pretrained_path', type=str, default='models/polyp/baseline_100epoch/best.pth')
    opt = parser.parse_args()
    
    train_path = '/home/U116med/data/polyp/TrainDataset/'
    image_root = '{}/images/'.format(train_path)
    gt_root = '{}/masks/'.format(train_path)
    
    # 建立 DataLoader
    train_loader = get_loader(image_root, gt_root, batchsize=16, trainsize=352)
    
    # 載入預訓練的 PVT_CASCADE 模型
    model = PVT_CASCADE(n_class=1, encoder='pvt_v2_b2').cuda()
    
    if os.path.exists(opt.pretrained_path):
        model.load_state_dict(torch.load(opt.pretrained_path))
        print(f"成功載入 Backbone 權重: {opt.pretrained_path}")
    else:
        print(f"警告: 找不到權重檔 {opt.pretrained_path}，將使用預設初始化提取。")
    
    # 執行特徵提取與分群
    prototypes, ch_dim = extract_and_cluster_features(
        model=model, 
        dataloader=train_loader, 
        target_node=opt.node,
        num_classes=2, 
        num_prototype=5
    )
    
    # 儲存 Prototype 權重 (檔名自動加上節點名稱與維度，避免混淆)
    save_path = f'kmeans_prototypes_{opt.node}_ch{ch_dim}.pth'
    torch.save(prototypes, save_path)
    print(f"Prototype 初始化權重已儲存至 {save_path}，形狀為: {prototypes.shape}")