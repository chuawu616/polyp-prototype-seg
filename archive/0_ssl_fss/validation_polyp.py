import os
import glob
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
from models.grid_proto_fewshot import FewShotSeg
from util.metric import Metric
from config_ssl_polyp import ex
from util.utils import set_seed
# --- 驗證資料集定義 ---
class PolypValidationDataset(Dataset):
    """
    專為驗證設計的資料集類別。
    它負責讀取影像和對應的真實遮罩，並將它們統一到固定的尺寸。
    """
    def __init__(self, image_dir, mask_dir, output_size=(256, 256)):
        """
        初始化驗證資料集。
        
        Args:
            image_dir (str): 驗證影像所在的資料夾路徑。
            mask_dir (str): 真實遮罩所在的資料夾路徑。
            output_size (tuple): 所有影像和遮罩將被統一調整到的目標尺寸。
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.output_size = output_size
        
        # 掃描影像資料夾，支援 .png 和 .jpg 格式
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, '*.png')))
        self.image_paths.extend(sorted(glob.glob(os.path.join(self.image_dir, '*.jpg'))))
        
        # 建立一個從影像檔名到遮罩檔案路徑的映射，以快速查找
        self.mask_map = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(os.path.join(self.mask_dir, '*.png'))
        }
        
        # 過濾掉那些沒有對應遮罩的影像，確保資料的完整性
        self.image_paths = [p for p in self.image_paths if os.path.splitext(os.path.basename(p))[0] in self.mask_map]

        if not self.image_paths:
            raise FileNotFoundError(f"在以下路徑中找不到任何匹配的 影像-遮罩 對：\n"
                                    f"  影像目錄: {self.image_dir}\n"
                                    f"  遮罩目錄: {self.mask_dir}")
        print(f"找到 {len(self.image_paths)} 組有效的驗證 影像-遮罩 對。")

        # 定義影像的轉換流程 (使用 torchvision.transforms)
        # 這是深度學習模型的標準預處理流程
        self.image_transform = T.Compose([
            T.Resize(self.output_size, interpolation=T.InterpolationMode.BILINEAR), # 雙線性插值調整尺寸
            T.ToTensor(), # 將 PIL Image 轉換為 PyTorch Tensor，並將像素值從 [0, 255] 縮放到 [0, 1]
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 使用ImageNet的均值和標準差進行Z-score標準化
        ])
        
        # 定義遮罩的轉換流程
        self.mask_transform = T.Compose([
            T.Resize(self.output_size, interpolation=T.InterpolationMode.NEAREST), # 必須使用最近鄰插值，以保護遮罩的類別索引值
        ])

    def __len__(self):
        """返回資料集的總樣本數。"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        根據索引 `idx` 獲取一個資料樣本。
        """
        image_path = self.image_paths[idx]
        # 使用 PIL 讀取影像，因為這是 torchvision transforms 的標準輸入格式
        img_pil = Image.open(image_path).convert("RGB")
        
        # 根據影像檔名找到對應的遮罩路徑
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = self.mask_map[base_name]
        mask_pil = Image.open(mask_path).convert("L") # 以灰度模式讀取遮罩

        # 應用各自的轉換流程
        img_tensor = self.image_transform(img_pil)
        
        # 對遮罩進行特殊處理，因為 Resize 後還需要二值化
        mask_resized_pil = self.mask_transform(mask_pil)
        mask_np = np.array(mask_resized_pil)
        mask_binary = (mask_np > 128).astype(np.uint8) # 將灰度圖轉換為 0 和 1 的二元遮罩
        mask_tensor = torch.from_numpy(mask_binary).long() # 轉換為長整數型的 Tensor
        
        # 額外返回未經處理的原始影像，用於後續的視覺化儲存
        original_image_np = np.array(img_pil)

        return {'image': img_tensor, 'mask': mask_tensor, 'id': idx, 'path': image_path, 'original_image': original_image_np}


def save_segmentation_results(original_image, pred_mask, true_mask, output_path):
    """
    將原圖、預測遮罩和真實遮罩拼接並儲存。
    """
    BG_COLOR_RGB = [0, 0, 0]
    FG_COLOR_RGB = [255, 255, 0] # RGB for Yellow

    if original_image.dtype != np.uint8:
        original_image = (original_image * 255).clip(0, 255).astype(np.uint8)

    # 獲取原始圖像的尺寸
    h, w, _ = original_image.shape

    # pred_mask (来自模型) 的尺寸是 256x256
    # true_mask (来自Dataset) 的尺寸也是 256x256
    # 我们需要将它们都放大回 h x w
    # 使用 cv2.INTER_NEAREST (最近邻插值) 来保持遮罩的 0/1 属性
    pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    true_mask_resized = cv2.resize(true_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    # 創建一個與原始圖像尺寸相同的三通道彩色圖像
    pred_colored = np.zeros((h, w, 3), dtype=np.uint8)
    true_colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 現在使用 resize 過的遮罩进行布林索引
    pred_colored[pred_mask_resized == 1] = FG_COLOR_RGB
    pred_colored[pred_mask_resized == 0] = BG_COLOR_RGB
    
    true_colored[true_mask_resized == 1] = FG_COLOR_RGB
    true_colored[true_mask_resized == 0] = BG_COLOR_RGB

    font = cv2.FONT_HERSHEY_SIMPLEX
    def add_title(image, text):
        img_with_title = cv2.copyMakeBorder(image, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.putText(img_with_title, text, (10, 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        return img_with_title

    original_with_title = add_title(original_image, 'Original Image')
    pred_with_title = add_title(pred_colored, 'Prediction')
    true_with_title = add_title(true_colored, 'Ground Truth')

    combined_image_rgb = np.concatenate((original_with_title, pred_with_title, true_with_title), axis=1)

    combined_image_bgr = cv2.cvtColor(combined_image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, combined_image_bgr)


# --- Sacred 實驗主體 ---
@ex.automain
def main(_run, _config, _log):
    # --- 環境與隨機種子設定 ---
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    set_seed(_config['seed']) # 確保隨機抽樣可複現

    # --- 載入已訓練好的模型 ---
    _log.info(f'###### 正在從 {_config["reload_model_path"]} 載入模型 ######')
    model = FewShotSeg(cfg=_config['model_cfg']).cuda()
    model.load_state_dict(torch.load(_config['reload_model_path'], weights_only=True))
    model.eval() # 將模型設置為評估模式

    # --- 建立驗證資料集和DataLoader ---
    _log.info('###### 正在載入驗證資料 ######')
    val_cfg = _config['validation_cfg']
    # 實例化我們修改後的資料集類別
    val_dataset = PolypValidationDataset(
        image_dir=val_cfg['image_dir'],
        mask_dir=val_cfg['mask_dir']
    )
    # 批次大小必須為1，因為我們手動模擬Few-shot場景
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)

    # --- 初始化評估指標計算器 ---
    metric = Metric(max_label=1) # max_label=1 因為只有背景(0)和息肉(1)
    n_shots = val_cfg['n_shots']

    # --- 設定視覺化結果的儲存路徑並隨機選擇要儲存的圖像 ---
    num_images_to_save = 10
    if _run.observers: # 如果使用 Sacred 運行，儲存在實驗日誌目錄下
        output_dir = os.path.join(_run.observers[0].dir, 'segmentation_outputs')
    else: # 否則儲存在當前目錄下
        output_dir = './segmentation_outputs'
    os.makedirs(output_dir, exist_ok=True)
    _log.info(f"分割結果圖像將儲存至: {output_dir}")

    # 從驗證集中隨機選取 N 個索引，用於後續的視覺化儲存
    indices_to_save = random.sample(range(len(val_dataset)), k=min(num_images_to_save, len(val_dataset)))
    _log.info(f"將為以下索引的圖像儲存視覺化結果: {indices_to_save}")

    # --- 驗證迴圈 ---
    _log.info(f'###### 開始 {n_shots}-shot 驗證 ######')
    with torch.no_grad():
        support_pool_indices = random.sample(range(len(val_dataset)), k=min(len(val_dataset), 50))
        
        for i, query_sample in enumerate(val_loader):
            query_image = [query_sample['image'].cuda()]
            query_mask_np = query_sample['mask'].numpy().squeeze()
            query_id = query_sample['id'].item()
            original_image_np = query_sample['original_image'].numpy().squeeze()
            
            # --- Support Set 準備 ---
            available_supports = [idx for idx in support_pool_indices if idx != query_id]
            if len(available_supports) < n_shots:
                _log.warning(f"沒有足夠的支援樣本給查詢圖像 {query_id}。跳過。")
                continue
            support_indices = random.sample(available_supports, k=n_shots)
            
            support_images_list, support_fg_masks_list, support_bg_masks_list = [], [], []
            for s_idx in support_indices:
                support_sample = val_dataset[s_idx]
                s_img = support_sample['image'].unsqueeze(0).cuda()
                s_mask = support_sample['mask'].unsqueeze(0).cuda()
                
                support_images_list.append(s_img)
                support_fg_masks_list.append(s_mask.float())
                support_bg_masks_list.append((1 - s_mask).float()) # 背景遮罩也要转换
            
            # 将支援样本打包成模型期望的巢状列表格式
            support_images = [support_images_list]
            support_fg_masks = [support_fg_masks_list]
            support_bg_masks = [support_bg_masks_list]
            
            # --- 模型預測 ---
            pred, _, _, _ = model(support_images, support_fg_masks, support_bg_masks, query_image)
            # 在類別維度上取 argmax 得到最終的分割圖
            pred_mask_np = pred.argmax(dim=1).squeeze().cpu().numpy()
            
            # --- 記錄評估指標 ---
            metric.record(pred=pred_mask_np, target=query_mask_np, labels=[1])

            # --- 檢查當前索引是否需要儲存視覺化圖像 ---
            if query_id in indices_to_save:
                image_basename = os.path.basename(query_sample['path'][0])
                output_path = os.path.join(output_dir, f"result_{image_basename}")
                
                # 調用輔助函式來生成並儲存對比圖
                save_segmentation_results(
                    original_image=original_image_np,
                    pred_mask=pred_mask_np,
                    true_mask=query_mask_np,
                    output_path=output_path
                )
                _log.info(f"已為索引 {query_id} 儲存分割結果至 {output_path}")

            if (i + 1) % 50 == 0:
                _log.info(f"已驗證 {i + 1}/{len(val_loader)} 張圖像...")

    # --- 計算並打印最終結果 ---
    # --- 修改點：正确解包 get_mDice 的返回值 ---
    # get_mDice 返回 (class_dice_mean, class_dice_std, mean_dice_mean, mean_dice_std)
    # 我们主要关心第一个值：按类别计算的平均 Dice
    all_metrics = metric.get_mDice(labels=[1])
    class_dice_mean = all_metrics[0]
    overall_mean_dice = all_metrics[2]
    
    _log.info('驗證完成。')
    # class_dice_mean 是一个数组，因为我们只评估一个类别，所以取第一个元素
    _log.info(f'在 {len(val_loader)} 張查詢圖像上，息肉類別 (class 1) 的平均 Dice 分數為: {class_dice_mean[0]:.4f}')
    _log.info(f'所有掃描的平均 Dice 分數為: {overall_mean_dice:.4f}')
    
    # 将结果记录到 Sacred 日誌
    _run.log_scalar('validation_class_dice', class_dice_mean[0])
    _run.log_scalar('validation_overall_dice', overall_mean_dice)
    
    return class_dice_mean[0]

# --- 輔助函式: 設定隨機種子以保證實驗可複現性 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)