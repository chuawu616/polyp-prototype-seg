import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# --- 導入我們更新後的模型和數據加載器 ---
from models.PixelContrastiveModel import PixelContrastiveModel
from dataloaders.ContrastiveDataset import ContrastiveDataset

# --- InfoNCE 損失函數  ---
class PixelInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1, negative_samples=256):
        super(PixelInfoNCELoss, self).__init__()
        self.temperature = temperature
        self.negative_samples = negative_samples
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, z1, z2): # 現在輸入的是投影特徵 z
        B, C, H, W = z1.shape
        z1, z2 = F.normalize(z1, p=2, dim=1), F.normalize(z2, p=2, dim=1)
        z1_flat, z2_flat = z1.view(B, C, -1), z2.view(B, C, -1)
        positive_logits = (z1_flat * z2_flat).sum(1) / self.temperature
        neg_indices = torch.randint(0, H * W, (self.negative_samples,))
        negative_samples = z2_flat.permute(0, 2, 1)[:, neg_indices, :]
        negative_logits = torch.matmul(z1_flat.permute(0, 2, 1), negative_samples.permute(0, 2, 1)) / self.temperature
        logits = torch.cat([positive_logits.unsqueeze(-1), negative_logits], dim=-1)
        labels = torch.zeros(B * H * W, dtype=torch.long, device=z1.device)
        loss = self.criterion(logits.view(-1, 1 + self.negative_samples), labels)
        return loss

# --- 主訓練流程 ---
def train_stage1(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 數據 (使用增強版 Dataset)
    dataset = ContrastiveDataset(image_dir=cfg['image_dir'], image_size=(cfg['image_size'], cfg['image_size']))
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    
    # 2. 模型 (實例化帶投影頭的模型)
    model = PixelContrastiveModel(
        pretrained=True, 
        target_stages=cfg['target_stages'],
        proj_dim=cfg['proj_dim']
    ).to(device)
    
    reload_path = cfg.get('reload_path', None)
    if reload_path and os.path.exists(reload_path):
        print(f"###### 正在從 {reload_path} 加載預訓練的 Encoder 權重... ######")
        try:
            # 加載 state_dict
            state_dict = torch.load(reload_path, map_location=device)
            
            # **重要**: 我們保存的是 encoder.state_dict()，所以可以直接加載到 model.encoder
            missing_keys, unexpected_keys = model.encoder.load_state_dict(state_dict, strict=True)
            print("###### Encoder 權重加載成功! ######")
            if missing_keys: print(f"警告: 模型中有一些權重未被加載: {missing_keys}")
            if unexpected_keys: print(f"警告: 權重文件中有一些未被使用的鍵: {unexpected_keys}")

        except Exception as e:
            print(f"!!!!!! 加載權重失敗: {e} !!!!!!")
            print("!!!!!! 將使用隨機初始化的權重進行訓練 !!!!!!")
    else:
        print("###### 未提供有效的權重路徑，將使用隨機初始化的權重進行訓練。######")
        model.encoder = pvt_v2_b2(pretrained=True)
        model.to(device)
        
    # 3. 損失函數與優化器 
    criterion = PixelInfoNCELoss(temperature=cfg['temperature'], negative_samples=cfg['negative_samples'])
    optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    # 4. 訓練迴圈
    print("###### 開始第一階段預訓練 (帶投影頭) ######")
    for epoch in range(cfg['epochs']):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        for view1, view2 in progress_bar:
            view1, view2 = view1.to(device), view2.to(device)

            optimizer.zero_grad()
            
            # 獲取【投影後】的多尺度特征 z
            projections1_list = model(view1)
            projections2_list = model(view2)
            
            # 在每個目標 stage 的投影特征 z 上計算損失並累加
            total_loss = 0
            for z1, z2 in zip(projections1_list, projections2_list):
                total_loss += criterion(z1, z2)
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            progress_bar.set_postfix(loss=f'{total_loss.item():.4f}')
            
        scheduler.step()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} 完成, 平均損失: {avg_epoch_loss:.4f}")

        # --- 核心修改點：儲存模型時，只保存 Encoder 的權重 ---
        if (epoch + 1) % cfg['save_every_epochs'] == 0:
            os.makedirs(cfg['output_dir'], exist_ok=True)
            save_path = os.path.join(cfg['output_dir'], f"stage1_epoch_{epoch+1}.pth")
            
            # **關鍵**: 我們只對下遊任務有用的 Encoder 進行保存
            # 我們丟棄 projection_heads 的權重
            torch.save(model.encoder.state_dict(), save_path)
            print(f"已儲存【Encoder】的權重快照至: {save_path}")

if __name__ == '__main__':
    # === 場景A: 在 Test Set 上微調 ===
    config_finetune_on_test = {
        'image_dir': "/home/U116med/data/polyp/TestDataset/test/images/", 
        
        'reload_path': './pretrained_stage1_proj/stage1_epoch_40.pth',
        
        'image_size': 352,
        'batch_size': 4,
        'epochs': 20, # 在新數據集上微調，通常不需要很長的 epoch
        
        'target_stages': [0, 1, 2, 3],
        'proj_dim': 128,
        
        'temperature': 0.1,
        'negative_samples': 256,
        

        'lr': 1e-5, # 比從頭訓練的 1e-4 小一個數量級
        'weight_decay': 1e-2,
        
        # 使用新的輸出目錄以區分
        'output_dir': './pretrained_stage1_finetuned_on_test/',
        'save_every_epochs': 10,
    }

    # === 場景B: 像以前一樣，在 Train Set 上從頭訓練 ===
    config_train_from_scratch = {
        'image_dir': '/home/U116med/data/polyp/TrainDataset/images/', 
        'reload_path': './models/pvt_v2_b2.pth',
        
        'image_size': 352,
        'batch_size': 4,
        'epochs': 40,
        
        'target_stages': [0, 1, 2, 3],
        'proj_dim': 128,
        
        'temperature': 0.1,
        'negative_samples': 256,
        
        'lr': 1e-4,
        'weight_decay': 1e-2,
        
        'output_dir': './pretrained_stage1_proj/',
        'save_every_epochs': 20,
    }
    # train_stage1(config_train_from_scratch)
    train_stage1(config_finetune_on_test)