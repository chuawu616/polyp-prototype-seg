import os
import itertools
import glob
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'
ex = Experiment('polypSSL')
ex.captured_out_filter = apply_backspaces_and_linefeeds

source_folders = ['.', './dataloaders', './models', './util']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]))
for source_file in sources_to_save:
    ex.add_source_file(source_file)

@ex.config
def cfg():
    """Default configurations for 2D Polyp SSL Training"""
    # --- 基礎設定 ---
    seed = 1234
    gpu_id = 0
    num_workers = 4  # 設為0有助於調試
    exp_prefix = 'PolypSSL_Run'

    # --- 資料集設定 ---
    dataset = 'POLYP'
    # 訓練影像的根目錄
    train_image_dir = "/home/U116med/data/polyp/TrainDataset/images/"
    # 訓練偽標籤的根目錄
    train_pseudolabel_dir = "/home/U116med/wch_code/ssl_fss/data/polypdata/pseudolabels_felzenszwalb/"
    # 驗證影像的根目錄
    val_image_dir = "/home/U116med/data/polyp/TestDataset/test/images/"
    # 驗證遮罩的根目錄
    val_mask_dir = "/home/U116med/data/polyp/TestDataset/test/masks/"
    # 使用哪種資料增強策略
    which_aug = 'sabs_aug' # 可選 'sabs_aug' (較溫和) 或 'aug_v3' (較激進)
    
    # --- 訓練過程設定 ---
    n_steps = 20000
    batch_size = 8
    # 學習率調整策略
    lr_milestones = list(range(0, n_steps, 5000))[1:] # 每5000步衰減一次
    lr_step_gamma = 0.5
    # 訓練日誌和模型儲存頻率
    print_interval = 100
    save_snapshot_every = 10000
    min_area_threshold = 100 # 設置一個您認為合理的預設值
    
    # --- 模型設定 ---
    model_name = 'dlfcn_res101'
    use_coco_init = True
    reload_model_path = None
    
    # --- ALPNet 核心設定 ---
    proto_grid_size = 4      # 在特徵圖上產生 (H'/size) x (W'/size) 的局部原型網格
    feature_hw = (32, 32)    # 骨幹網路輸出的特徵圖尺寸，需與實際模型匹配
    use_align_loss = True
    align_loss_weight = 1.0   
    
    # --- 自監督任務設定 ---
    # `num_rep` 決定了每個樣本增強幾次來配對。
    # num_rep=2 -> 1-shot, 1-query.
    num_rep = 2
    boundary_loss_weight = 1.0
    
    # --- 優化器設定 ---
    optim_type = 'adamw'
    
    # 定义 AdamW 的专属参数
    adamw_params = {
        'lr': 1e-4,             # AdamW 的典型学习率
        'betas': (0.9, 0.999),  # AdamW 的标准 beta 参数
        'weight_decay': 0.01,   # AdamW 的典型权重衰减
    }

    # 定义 SGD with Momentum 的专属参数 ---
    sgd_params = {
        'lr': 1e-3,             # SGD 的典型学习率，通常比 AdamW 大
        'momentum': 0.9,        # SGD+Momentum 的标准动量值
        'weight_decay': 5e-4,   # SGD 的典型权重衰减
    }

    # --- 逻辑判断：根据 optim_type 选择最终的 optim_params ---
    # 这个 optim_params 将被传递给 training_polyp.py
    if optim_type == 'adamw':
        optim_params = adamw_params
    elif optim_type == 'sgd':
        optim_params = sgd_params
    else:
        raise ValueError(f"Unsupported optimizer type: {optim_type}")

    # --- 將設定整理成字典，方便傳遞給各個模組 ---
    model_cfg = {
        'align': use_align_loss,
        'use_coco_init': use_coco_init,
        'which_model': model_name,
        'cls_name': 'grid_proto', # 固定為grid_proto
        'proto_grid_size': proto_grid_size,
        'feature_hw': feature_hw,
        'reload_model_path': reload_model_path
    }
    
    dataloader_cfg = {
        'image_dir': train_image_dir,
        'pseudolabel_dir': train_pseudolabel_dir,
        'aug_name': which_aug,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'num_rep': num_rep,
        'min_area_threshold': min_area_threshold
    }
    
    # --- 清理後的驗證設定 ---
    validation_cfg = {
        'image_dir': val_image_dir,
        'mask_dir': val_mask_dir,
        'n_shots': 1
    }
    
    # --- 實驗名稱 ---
    exp_str = '_'.join(
        [exp_prefix, dataset, f'grid{proto_grid_size}', f'bs{batch_size}', f"lr{optim_params['lr']}"])

    path = {
        'log_dir': './runs'
    }

@ex.config_hook
def add_observer(config, command_name, logger):
    """A hook function to add observer"""
    exp_name = f'{ex.path}_{config["exp_str"]}'
    observer = FileStorageObserver.create(os.path.join(config['path']['log_dir'], exp_name))
    ex.observers.append(observer)
    return config