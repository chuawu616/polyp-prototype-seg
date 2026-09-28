import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

from lib.networks_prototype_specialize import DINOv3_Prototype_Specialize
from lib.networks_prototype_concat import DINOv3_Prototype_concat 
from lib.networks_prototype_specialize_pvtv2 import Prototype_CASCADE_Specialize
from lib.networks_prototype_pvtv2 import Prototype_CASCADE
from utils.dataloader import test_dataset

def evaluate_pixel_metrics(model, loader):
    """計算全資料集的 TP, TN, FP, FN 像素總數"""
    metrics = {
        'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0
    }
    
    num_images = loader.size
    
    with torch.no_grad():
        for i in tqdm(range(num_images), desc="推論中"):
            image_tensor, gt_mask, name = loader.load_data()
            
            # GT 二值化處理
            gt_np = np.asarray(gt_mask, np.float32)
            gt_np /= (gt_np.max() + 1e-8)
            gt_binary = np.where(gt_np >= 0.5, 1, 0).astype(bool)
            
            image_tensor = image_tensor.cuda()
            H, W = gt_binary.shape

            # 模型推論
            logits_binary, _ = model(image_tensor)
            
            # 二元預測處理
            res = (logits_binary[:, 1:2, :, :] - logits_binary[:, 0:1, :, :]) * 10.0
            res = F.interpolate(res, size=(H, W), mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            pred_binary = np.where(res >= 0.5, 1, 0).astype(bool)
            
            # --- 像素級混淆矩陣計算 ---
            metrics['TP'] += np.sum(pred_binary & gt_binary)
            metrics['TN'] += np.sum(~pred_binary & ~gt_binary)
            metrics['FP'] += np.sum(pred_binary & ~gt_binary)
            metrics['FN'] += np.sum(~pred_binary & gt_binary)

    return metrics

def calculate_derived_metrics(m):
    """根據 TP, TN, FP, FN 計算進階指標"""
    TP, TN, FP, FN = m['TP'], m['TN'], m['FP'], m['FN']
    
    # 避免分母為 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0  # 偽陽性率 (越低越好)
    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0.0  # 偽陰性率 (越低越好)
    
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0 # 召回率 (越高越好)
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0 # 精確率 (越高越好)
    
    Dice = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
    
    return {
        'FPR': FPR, 'FNR': FNR, 
        'Recall': Recall, 'Precision': Precision, 'Dice': Dice
    }

def main():
    parser = argparse.ArgumentParser(description="Compare FP and FN between two Prototype Models")
    parser.add_argument('--pth_a', type=str, 
                        default='/home/U116med/wch_code/cascade/models/prototype_cascade/baseline_prototype_fhb_424_warm_10/best.pth', 
                        help='模型 A (Specialize) best.pth 路徑')
    parser.add_argument('--pth_b', type=str, 
                        default='/home/U116med/wch_code/cascade/models/prototype_cascade/baseline_prototype_fb_44/best.pth', 
                        help='模型 B (Baseline/Concat) best.pth 路徑')
    
    parser.add_argument('--data_path', type=str, default='/home/U116med/data/polyp/TestDataset/test', help='測試資料集路徑')
    parser.add_argument('--backbone', type=str, default='vits16plus', help='DINOv3 版本')
    parser.add_argument('--testsize', type=int, default=352, help='推論尺寸')
    parser.add_argument('--num_fg', type=int, default=4, help='前景原型數量')
    parser.add_argument('--num_hard', type=int, default=2, help='困難原型數量')
    parser.add_argument('--num_bg', type=int, default=4, help='背景原型數量')
    opt = parser.parse_args()

    image_root = os.path.join(opt.data_path, 'images/')
    gt_root = os.path.join(opt.data_path, 'masks/')
    dataset_name = os.path.basename(os.path.normpath(opt.data_path))

    # --- 模型 A 評估 ---
    test_loader_a = test_dataset(image_root, gt_root, opt.testsize)
    # model_a = DINOv3_Prototype_Specialize(num_fg=opt.num_fg, num_hard=opt.num_hard, num_bg=opt.num_bg, backbone_type='vits16plus').cuda()
    model_a = Prototype_CASCADE_Specialize(num_fg=opt.num_fg, num_hard=opt.num_hard, num_bg=opt.num_bg).cuda()
    
    if os.path.exists(opt.pth_a):
        print(f"Loading weights from: {opt.pth_a}")
        checkpoint_a = torch.load(opt.pth_a, map_location='cuda', weights_only=True)
        model_a.load_state_dict(checkpoint_a, strict=True)
    else:
        print('pth_a does not exist')
    
    model_a.eval()
    print(f"\n[1/2] 評估模型 A (Specialize)...")
    raw_metrics_a = evaluate_pixel_metrics(model_a, test_loader_a)
    der_metrics_a = calculate_derived_metrics(raw_metrics_a)
    
    # --- 模型 B 評估 ---
    test_loader_b = test_dataset(image_root, gt_root, opt.testsize) 
    # model_b = DINOv3_Prototype_concat(num_fg=opt.num_fg, num_bg=opt.num_bg, backbone_type='vits16plus').cuda()
    model_b = Prototype_CASCADE(num_fg=opt.num_fg, num_bg=opt.num_bg).cuda()
    if os.path.exists(opt.pth_b):
        print(f"Loading weights from: {opt.pth_b}")
        checkpoint_b = torch.load(opt.pth_b, map_location='cuda', weights_only=True)
        model_b.load_state_dict(checkpoint_b, strict=True)
    else:
        print('pth_b does not exist')
    model_b.eval()
    print(f"\n[2/2] 評估模型 B (Baseline)...")
    raw_metrics_b = evaluate_pixel_metrics(model_b, test_loader_b)
    der_metrics_b = calculate_derived_metrics(raw_metrics_b)

    output_filename = f"comparison_report_{dataset_name}.txt"

    # --- 開始寫入檔案與顯示結果 ---
    with open(output_filename, 'w', encoding='utf-8') as f:
        def print_both(text):
            print(text)      # 顯示在螢幕
            f.write(text + "\n") # 寫入檔案

        print_both("\n" + "="*70)
        print_both(f"像素級 False Positive / False Negative 對比報告")
        print_both(f"資料集: {dataset_name} | 圖片數: {test_loader_a.size}")
        print_both("="*70)
        
        # 修改 format_diff 移除 ANSI 顏色代碼 (避免 txt 出現亂碼)
        def format_diff_txt(val_a, val_b, lower_is_better=True):
            diff = val_a - val_b
            pct_change = (diff / (val_b + 1e-8)) * 100
            if diff == 0: return "無變化"
            is_improved = (diff < 0) if lower_is_better else (diff > 0)
            arrow = "↓" if diff < 0 else "↑"
            status = "[改善]" if is_improved else "[惡化]"
            return f"{arrow} {abs(diff):.4f} ({pct_change:+.2f}%) {status}"

        # 準備表格數據 (使用不帶顏色的格式化)
        table_data = [
            ["指標 (Metric)", "模型 A (Specialize)", "模型 B (Baseline)", "A vs B 差異"],
            ["--- 絕對像素數量 ---", "", "", ""],
            ["FP 像素", f"{raw_metrics_a['FP']:,}", f"{raw_metrics_b['FP']:,}", format_diff_txt(raw_metrics_a['FP'], raw_metrics_b['FP'], True)],
            ["FN 像素", f"{raw_metrics_a['FN']:,}", f"{raw_metrics_b['FN']:,}", format_diff_txt(raw_metrics_a['FN'], raw_metrics_b['FN'], True)],
            ["TP 像素", f"{raw_metrics_a['TP']:,}", f"{raw_metrics_b['TP']:,}", format_diff_txt(raw_metrics_a['TP'], raw_metrics_b['TP'], False)],
            ["TN 像素", f"{raw_metrics_a['TN']:,}", f"{raw_metrics_b['TN']:,}", format_diff_txt(raw_metrics_a['TN'], raw_metrics_b['TN'], False)],
            ["--- 關鍵比率指標 ---", "", "", ""],
            ["FPR (↓)", f"{der_metrics_a['FPR']:.5f}", f"{der_metrics_b['FPR']:.5f}", format_diff_txt(der_metrics_a['FPR'], der_metrics_b['FPR'], True)],
            ["FNR (↓)", f"{der_metrics_a['FNR']:.5f}", f"{der_metrics_b['FNR']:.5f}", format_diff_txt(der_metrics_a['FNR'], der_metrics_b['FNR'], True)],
            
            ["Precision (↑)", f"{der_metrics_a['Precision']:.4f}", f"{der_metrics_b['Precision']:.4f}", 
             format_diff_txt(der_metrics_a['Precision'], der_metrics_b['Precision'], False)],
            
            ["Recall (↑)", f"{der_metrics_a['Recall']:.4f}", f"{der_metrics_b['Recall']:.4f}", 
             format_diff_txt(der_metrics_a['Recall'], der_metrics_b['Recall'], False)],
            
            ["Overall DICE (↑)", f"{der_metrics_a['Dice']:.4f}", f"{der_metrics_b['Dice']:.4f}", 
             format_diff_txt(der_metrics_a['Dice'], der_metrics_b['Dice'], False)],
        ]
        
        # 輸出表格
        formatted_table = tabulate(table_data, headers="firstrow", tablefmt="fancy_grid")
        print_both(formatted_table)
        
        # 自動診斷結論
        print_both("\n[架構診斷分析]")
        fp_diff = raw_metrics_a['FP'] - raw_metrics_b['FP']
        fn_diff = raw_metrics_a['FN'] - raw_metrics_b['FN']
        
        if fp_diff < 0 and fn_diff < 0:
            msg = "完美提升：Specialize 模型同時降低了偽陽性 (FP) 與偽陰性 (FN)。"
        elif fp_diff < 0 and fn_diff > 0:
            msg = "權衡現象 (更保守)：Specialize 模型成功降低了 FP，但代價是漏判 (FN) 增加。"
        elif fp_diff > 0 and fn_diff < 0:
            msg = "權衡現象 (更激進)：Specialize 模型的 FN 降低了，但 FP 增加。"
        else:
            msg = "全面惡化：Specialize 模型的 FP 和 FN 都比 Baseline 差。"
        
        print_both(msg)
        print_both("="*70 + "\n")
        
    print(f"報告已成功儲存至: {output_filename}")
if __name__ == '__main__':
    main()