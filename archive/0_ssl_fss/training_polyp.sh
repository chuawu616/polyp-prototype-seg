#!/bin/bash

GPUID=0
export CUDA_VISIBLE_DEVICES=$GPUID


CPT="PolypSSL_Experiment1"


TRAIN_IMG_DIR="/home/U116med/data/polyp/TrainDataset/images/"
TRAIN_PSEUDO_DIR="/home/U116med/wch_code/ssl_fss/pvt_kmeans/polypdata/pvt_kmeans_masks_stage1_256"


N_STEPS=10000
BATCH_SIZE=8
LEARNING_RATE=1e-3
OPTIMIZER='adamw'
AUG_STRATEGY='sabs_aug'

PROTO_GRID_SIZE=8
MIN_AREA=100

DATASET='slic_kmeans'
LOG_ROOT_DIR="./runs"
SAVE_INTERVAL=5000
PRINT_INTERVAL=100
USE_ALIGN_LOSS=True
BOUNDARY_LOSS_WEIGHT=1.0


EXP_STR="bs${BATCH_SIZE}_lr${LEARNING_RATE}_grid${PROTO_GRID_SIZE}_min_area_${MIN_AREA}_${AUG_STRATEGY}_dataset_${DATASET}_opt_${OPTIMIZER}"
LOGDIR="${LOG_ROOT_DIR}/${CPT}/${EXP_STR}"

echo "========================================================"
echo "           TRAINING START"
echo "========================================================"
echo "Experiment: ${CPT}"
echo "LOG_DIR: ${LOGDIR}"
echo "IMG_DIR: ${TRAIN_IMG_DIR}"
echo "PSEUDO_DIR: ${TRAIN_PSEUDO_DIR}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "lr: ${LEARNING_RATE}"
echo "Total steps: ${N_STEPS}"
echo "========================================================"

python3 training_polyp.py with \
    exp_prefix="${CPT}" \
    n_steps=${N_STEPS} \
    batch_size=${BATCH_SIZE} \
    which_aug="${AUG_STRATEGY}" \
    save_snapshot_every=${SAVE_INTERVAL} \
    print_interval=${PRINT_INTERVAL} \
    boundary_loss_weight=${BOUNDARY_LOSS_WEIGHT} \
    "dataloader_cfg.min_area_threshold"=${MIN_AREA} \
    'model_cfg.proto_grid_size'=${PROTO_GRID_SIZE} \
    'model_cfg.align'=${USE_ALIGN_LOSS} \
    optim_type="${OPTIMIZER}" \
    "optim_params.lr"=${LEARNING_RATE} \
    "path.log_dir"="${LOG_ROOT_DIR}" \
    exp_str="${EXP_STR}" \
    "dataloader_cfg.image_dir"="${TRAIN_IMG_DIR}" \
    "dataloader_cfg.pseudolabel_dir"="${TRAIN_PSEUDO_DIR}"

echo "========================================================"
echo "           TRAINING Finish"
echo "========================================================"