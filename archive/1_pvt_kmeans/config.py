import os
import torch
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver

# --- Sacred 實驗設定 ---
# 建議為您的新專案取一個清晰的實驗名稱
ex = Experiment("Polyp_Prototype_Segmentation")

@ex.config
def cfg():
    """實驗的預設配置"""
    # --- 通用設定 ---
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 352

    # --- 數據路徑 ---
    train_image_dir = '/home/U116med/data/polyp/TrainDataset/images/'
    train_mask_dir = '/home/U116med/data/polyp/TrainDataset/masks/'
    val_image_dir = '/home/U116med/data/polyp/TestDataset/test/images/'
    val_mask_dir = '/home/U116med/data/polyp/TestDataset/test/masks/'
    train_prototype_dir = './all_protos_8_8/'
    val_prototype_dir = './valid_protos_8_8/'
    
    # --- 訓練參數 ---
    epochs = 100
    batch_size = 4
    num_workers = 4

    # --- 驗證參數 ---
    # 指定用於驗證的模型權重路徑。在運行驗證腳本時，通常會從命令列覆蓋此路徑。
    reload_model_path = './runs/pvtv2_fpn_proto_lr1e-05/2/checkpoint_epoch_10.pth'
    val_batch_size = 16
    eval_every_epochs = 1

    # --- 模型配置 (Model Configuration) ---
    model_cfg = {
        # (可選) 指定本地 PVTv2 預訓練權重的路徑
        # 如果設為 None 或檔案不存在，模型會嘗試從在線加載 ImageNet 權重
        'encoder_weight_path': './models/pvt_v2_b2.pth',

        # 原型相關設定
        'num_prototypes': 16,
        'prototype_path': './global_prototypes_128d.npy',
        'prototype_labels': [1]*8 + [0]*8, # 前8個為前景(1)，後8個為背景(0)
        # Decoder 相關設定
        'decoder_out_channels': 128,
        # 'freeze_prototypes': True,
    }
    
    # --- 優化器參數 (Optimizer Configuration) ---
    # 分層學習率設定
    optimizer_cfg = {
        'encoder_lr': 1e-4,
        'decoder_lr': 1e-4,
        # 'prototype_lr': 'freeze',
        'weight_decay': 0.01,
    }
    
    # --- 損失函數設定 ---
    loss_cfg = {
        'bce_weight': 0.5,
        'iou_weight': 0.5
    }

    # --- 實驗記錄 ---
    # 實驗的根目錄
    log_dir = './runs'
    # 實驗名稱，用於在 log_dir 下創建子資料夾
    exp_name = f"enlr{optimizer_cfg['encoder_lr']}_delr{optimizer_cfg['decoder_lr']}"

# --- Sacred Hook (用於自動創建日誌資料夾) ---
@ex.config_hook
def add_observer(config, command_name, logger):
    """一個鉤子函式，用於為每次運行添加檔案儲存觀察者。"""
    observer = FileStorageObserver(os.path.join(config['log_dir'], config['exp_name']))
    ex.observers.append(observer)
    return config