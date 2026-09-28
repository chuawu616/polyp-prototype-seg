import torch
from torch.utils.data._utils.collate import default_collate

def custom_collate_fn(batch):
    """
    自訂的 collate_fn，用於處理 few-shot 任務返回的巢狀 list 結構。

    Args:
        batch (list): 一個 list，其中每個元素都是 Dataset.__getitem__ 返回的字典。
    """
    # 1. 將 batch of dicts 轉換為 dict of lists
    # e.g., [{'img': T1}, {'img': T2}] -> {'img': [T1, T2]}
    collated_batch = {key: [d[key] for d in batch] for key in batch[0]}

    # 2. 處理特殊結構的鍵
    # `class_ids` 是一個 list of lists of ints，我們把它拍平
    # e.g., [[1], [1], [1], [1]] -> [1, 1, 1, 1] (如果都是1-way)
    collated_batch['class_ids'] = [item for sublist in collated_batch['class_ids'] for item in sublist]

    # `query_images` 和 `query_labels` 是 list of lists of Tensors
    # 我們將它們堆疊起來
    # [ [q1_1, q1_2], [q2_1, q2_2] ] -> [ torch.stack([q1_1, q2_1]), torch.stack([q1_2, q2_2]) ]
    # 假設每個樣本的 n_queries 相同
    n_queries = len(batch[0]['query_images'])
    collated_batch['query_images'] = [torch.stack([sample['query_images'][i] for sample in batch]) for i in range(n_queries)]
    collated_batch['query_labels'] = [torch.stack([sample['query_labels'][i] for sample in batch]) for i in range(n_queries)]
    
    # 3. 處理最複雜的 support set
    # support_images: list[batch_size][way(1)][shot]
    # 我們需要將其轉換為: [way(1)][shot][batch_size, C, H, W]
    n_ways = len(batch[0]['support_images'])
    n_shots = len(batch[0]['support_images'][0])

    collated_supports = []
    for way_idx in range(n_ways):
        shots_collated = []
        for shot_idx in range(n_shots):
            # 從 batch 中每個樣本的對應位置取出 shot tensor 並堆疊
            shot_tensor = torch.stack([sample['support_images'][way_idx][shot_idx] for sample in batch])
            shots_collated.append(shot_tensor)
        collated_supports.append(shots_collated)
    collated_batch['support_images'] = collated_supports

    # `support_mask` 是 list of list of dicts，我們保持其 list 結構
    # 因為它不是 tensor，default_collate 無法處理，我們手動組合
    # 結構: [way(1)][shot][list of batch_size dicts]
    collated_support_masks = []
    for way_idx in range(n_ways):
        shots_collated = []
        for shot_idx in range(n_shots):
            shot_masks = [sample['support_mask'][way_idx][shot_idx] for sample in batch]
            shots_collated.append(shot_masks)
        collated_support_masks.append(shots_collated)
    collated_batch['support_mask'] = collated_support_masks

    # 4. 其他鍵保持原樣 (通常是空的 list)
    collated_batch['support_inst'] = collated_batch['support_inst']
    collated_batch['support_scribbles'] = collated_batch['support_scribbles']
    
    return collated_batch