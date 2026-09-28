from torch.utils.data import DataLoader
from dataloaders.PolypSuperpixelDataset import PolypSuperpixelDataset
from dataloaders.augutils import get_geometric_transformer, get_intensity_transformer, transform_with_label, augs
from .custom_collate import custom_collate_fn
def build_ssl_dataloader(cfg):
    """
    Args:
        cfg (dict): 
            - image_dir (str): 影像資料夾路徑。
            - pseudolabel_dir (str): 偽標籤資料夾路徑。
            - aug_name (str): 資料增強策略名稱。
            ...
    """
    aug_settings = {'aug': augs[cfg['aug_name']]}
    transforms = transform_with_label(aug_settings)

    # --- 修改點：傳遞獨立路徑 ---
    dataset = PolypSuperpixelDataset(
        image_dir=cfg['image_dir'],
        pseudolabel_dir=cfg['pseudolabel_dir'],
        mode='train',
        transforms=transforms,
        num_rep=cfg['num_rep'],
        min_area_threshold=cfg.get('min_area_threshold', 100)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )

    return dataloader