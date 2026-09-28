import os
import shutil
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from models.grid_proto_fewshot import FewShotSeg
from dataloaders.dataloader_entry import build_ssl_dataloader 
from util.utils import set_seed
from util.losses import BoundaryLoss
from config_ssl_polyp import ex

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        log_dir = _run.observers[0].dir
        os.makedirs(os.path.join(log_dir, 'snapshots'), exist_ok=True)
    
    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    # --- 1. 建立模型 ---
    _log.info('###### Create model ######')
    model = FewShotSeg(cfg=_config['model_cfg']).cuda()
    model.train()

    # --- 2. 建立資料載入器 ---
    _log.info('###### Load data ######')
    trainloader = build_ssl_dataloader(_config['dataloader_cfg'])

    # --- 3. 設定優化器和損失函數 ---
    _log.info('###### Set optimizer ######')
    if _config['optim_type'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **_config['optim_params'])
    elif _config['optim_type'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), **_config['optim_params'])
    else:
        raise NotImplementedError
    
    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=_config['lr_step_gamma'])
    
    criterion_ce = nn.CrossEntropyLoss(ignore_index=255)
    criterion_boundary = BoundaryLoss(device=f'cuda:{_config["gpu_id"]}')
    
    # --- 4. 訓練迴圈  ---
    log_loss = {'loss': 0, 'ce_loss': 0, 'boundary_loss':0, 'align_loss': 0}
    i_iter = 0
    _log.info('###### Training ######')
    
    # 建立一個可重複使用的迭代器
    dataloader_iterator = iter(trainloader)
    
    # 使用 while 迴圈，直到達到 n_steps
    while i_iter < _config['n_steps']:
        
        try:
            sample_batched = next(dataloader_iterator)
        except StopIteration:
            # 當一個epoch結束時，重新創建迭代器
            dataloader_iterator = iter(trainloader)
            sample_batched = next(dataloader_iterator)

        # support_images 結構: [way[shot[B,C,H,W]]]
        # 我們需要將所有Tensor移動到GPU
        support_images = [[shot.cuda() for shot in way] 
                          for way in sample_batched['support_images']]
        
        # support_mask 結構: [way[shot[list of B dicts]]]
        # 我們需要遍歷這個結構，將每個dict中的Tensor移動到GPU
        support_fg_masks = []
        support_bg_masks = []
        for way_idx, way_shots in enumerate(sample_batched['support_mask']):
            fg_shots, bg_shots = [], []
            for shot_idx, shot_masks_list in enumerate(way_shots):
                # shot_masks_list 是一個包含 B 個 dict 的 list
                # 我們需要將它們轉換為 B 個 fg_mask tensor 和 B 個 bg_mask tensor，然後堆疊
                fg_mask_batch = torch.stack([m['fg_mask'] for m in shot_masks_list]).float().cuda()
                bg_mask_batch = torch.stack([m['bg_mask'] for m in shot_masks_list]).float().cuda()
                fg_shots.append(fg_mask_batch)
                bg_shots.append(bg_mask_batch)
            support_fg_masks.append(fg_shots)
            support_bg_masks.append(bg_shots)

        # Query 數據現在是 list of Tensors，每個 Tensor 已經是 batch 形式
        query_images = [query_image.cuda() 
                        for query_image in sample_batched['query_images']]
        query_labels = torch.cat(
            [query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)

        # --- 前向傳播和反向傳播  ---
        optimizer.zero_grad()
        
        try:
            query_pred, align_loss, _, _ = model(
                support_images, support_fg_masks, support_bg_masks, query_images)
        except Exception as e:
            _log.warning(f'Faulty batch detected at step {i_iter}, skipping. Error: {e}')
            continue

        # 1. 計算交叉熵損失
        ce_loss = criterion_ce(query_pred, query_labels)
        
        # 2. 計算邊界損失
        #    需要將模型的 logits 輸出轉換為概率
        query_probs = F.softmax(query_pred, dim=1)
        boundary_loss = criterion_boundary(query_probs, query_labels)
        
        # 3. 組合損失
        boundary_loss_weight = _config['boundary_loss_weight']
        loss = ce_loss + boundary_loss_weight * boundary_loss + _config['align_loss_weight'] * align_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        # --- 日誌記錄  ---
        log_loss['loss'] += loss.item()
        log_loss['ce_loss'] += ce_loss.item()
        log_loss['boundary_loss'] += boundary_loss.item()
        log_loss['align_loss'] += align_loss.item()
        
        if (i_iter + 1) % _config['print_interval'] == 0:
            ce_loss_val = log_loss['ce_loss'] / _config['print_interval']
            boundary_loss_val = log_loss['boundary_loss'] / _config['print_interval']
            align_loss_val = log_loss['align_loss'] / _config['print_interval']
            total_loss_val = log_loss['loss'] / _config['print_interval']
            
            log_loss['ce_loss'] = 0
            log_loss['boundary_loss'] = 0
            log_loss['loss'] = 0
            log_loss['align_loss'] = 0
            
            _log.info(f'step {i_iter+1}/{_config["n_steps"]}: total_loss: {total_loss_val:.4f}, ce_loss: {ce_loss_val:.4f}, boundary_loss: {boundary_loss_val:.4f}, align_loss: {align_loss_val:.4f}')
            _run.log_scalar('total_loss', total_loss_val, i_iter + 1)
            _run.log_scalar('ce_loss', ce_loss_val, i_iter + 1)
            _run.log_scalar('boundary_loss', boundary_loss_val, i_iter + 1)
            _run.log_scalar('align_loss', align_loss_val, i_iter + 1)

        # --- 儲存模型快照  ---
        if (i_iter + 1) % _config['save_snapshot_every'] == 0:
            _log.info(f'###### Taking snapshot at step {i_iter+1} ######')
            # 確保 log_dir 變數已定義
            log_dir = _run.observers[0].dir if _run.observers else 'runs/fallback'
            torch.save(model.state_dict(),
                       os.path.join(log_dir, 'snapshots', f'{i_iter + 1}.pth'))
        
        # 增加迭代計數器
        i_iter += 1

    _log.info('###### Training finished ######')