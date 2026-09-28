#!/bin/bash
# 腳本: validate_polyp.sh
# 描述: 驗證訓練好的模型在 Few-shot 息肉分割任務上的性能

# --- 1. 基礎設定 ---
GPUID=1
export CUDA_VISIBLE_DEVICES=$GPUID

# --- 2. 實驗參數 (在此處修改) ---

# **非常重要**: 填寫您訓練好的模型權重的路徑
RELOAD_MODEL_PATH="/home/U116med/wch_code/ssl_fss/runs/polypSSL_bs8_lr1e-3_grid8_min_area_100_sabs_aug_dataset_slic_kmeans_opt_adamw/1/snapshots/10000.pth"

# 驗證資料集的路徑 (使用您提供的路徑)
VAL_IMG_DIR="/home/U116med/data/polyp/TestDataset/test/images/"
VAL_MASK_DIR="/home/U116med/data/polyp/TestDataset/test/masks/"
# "/home/U116med/data/polyp/TestDataset/test/"
# "/home/U116med/data/polyp/TestDataset/CVC-300/"
# "/home/U116med/data/polyp/TestDataset/CVC-ClinicDB/"
# "/home/U116med/data/polyp/TestDataset/CVC-ColonDB/"
# "/home/U116med/data/polyp/TestDataset/ETIS-LaribPolypDB/"
# "/home/U116med/data/polyp/TestDataset/Kvasir/"

# 驗證時使用的 N-shot 設定
N_SHOTS=1 # 測試1-shot性能
#N_SHOTS=5 # 也可以改為5-shot

# 實驗名稱，用於日誌
CPT="PolypValidation_shot${N_SHOTS}"

# 日誌根目錄
LOG_ROOT_DIR="./runs_validation"

# 批次大小 (Batch Size)。
BATCH_SIZE=8

# --- 模型與優化器設定 ---
# 學習率 (Learning Rate)。
LEARNING_RATE=1e-3
# 優化器類型 ('sgd' 或 'adamw')。
OPTIMIZER='adamw'
# 數據增強策略 ('sabs_aug' 代表較溫和, 'aug_v3' 代表較激進)。
AUG_STRATEGY='sabs_aug'

# --- ALPNet 核心參數 ---
PROTO_GRID_SIZE=8
MIN_AREA=100
# --- 3. 組合實驗名稱和日誌目錄 ---
# 從模型路徑中提取模型步數，使日誌更具資訊性
# DATASET="felzenszwalb"
DATASET="slic_kmeans"
# DATASET="watershed"
# DATASET="supervised"

# VAL_DATASET="cvc-300"
# VAL_DATASET="kvasir"
# VAL_DATASET="cvc-clinicdb"
# VAL_DATASET="cvc-colondb"
VAL_DATASET="etis"


MODEL_STEP=$(basename "${RELOAD_MODEL_PATH}" .pth)
# EXP_STR="${DATASET}_${VAL_DATASET}/model_${MODEL_STEP}_bs${BATCH_SIZE}_lr${LEARNING_RATE}_grid${PROTO_GRID_SIZE}_${AUG_STRATEGY}_min_area_${MIN_AREA}"
EXP_STR="${DATASET}/model_${MODEL_STEP}_bs${BATCH_SIZE}_lr${LEARNING_RATE}_grid${PROTO_GRID_SIZE}_${AUG_STRATEGY}_min_area_${MIN_AREA}"
LOGDIR="${LOG_ROOT_DIR}/${CPT}/${EXP_STR}"

echo "========================================================"
echo "           Starting Polyp Validation"
echo "========================================================"
echo "Model Path: ${RELOAD_MODEL_PATH}"
echo "Validation Image Dir: ${VAL_IMG_DIR}"
echo "Validation Mask Dir: ${VAL_MASK_DIR}"
echo "Log Directory: ${LOGDIR}"
echo "N-Shots: ${N_SHOTS}"
echo "========================================================"

# 檢查模型路徑是否存在
if [ ! -f "$RELOAD_MODEL_PATH" ]; then
    echo "Error: Model path not found at ${RELOAD_MODEL_PATH}"
    exit 1
fi
# 檢查驗證資料夾是否存在
if [ ! -d "$VAL_IMG_DIR" ]; then
    echo "Error: Validation image directory not found at ${VAL_IMG_DIR}"
    exit 1
fi
if [ ! -d "$VAL_MASK_DIR" ]; then
    echo "Error: Validation mask directory not found at ${VAL_MASK_DIR}"
    exit 1
fi

# --- 4. 運行 Sacred 實驗 ---
# 使用 'with' 關鍵字傳遞參數來覆蓋 config_ssl_polyp.py 中的預設值
python3 validation_polyp.py with \
    exp_prefix="${CPT}" \
    reload_model_path="${RELOAD_MODEL_PATH}" \
    'model_cfg.proto_grid_size'=${PROTO_GRID_SIZE} \
    "path.log_dir"="${LOG_ROOT_DIR}" \
    exp_str="${EXP_STR}" \
    "validation_cfg.image_dir"="${VAL_IMG_DIR}" \
    "validation_cfg.mask_dir"="${VAL_MASK_DIR}" \
    "validation_cfg.n_shots"=${N_SHOTS}

echo "========================================================"
echo "           Validation Finished"
echo "========================================================"