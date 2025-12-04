#!/bin/bash

# 设置 wandb 项目名
export WANDB_PROJECT="TPGNN-hparam-tuning-easy"
# export WANDB_ENTITY="joella"  # 

# 数据路径
DATA_PATH="/mnt/webscistorage/cc7738/ws_joella/EnergyTSF_latest/EnergyTSF_20/TPGNN/datasets/synthetic_time_series1.csv"
STAMP_PATH="/mnt/webscistorage/cc7738/ws_joella/EnergyTSF_latest/EnergyTSF_20/TPGNN/datasets/synthetic_time_series1.npy"
ADJ_PATH="/mnt/webscistorage/cc7738/ws_joella/EnergyTSF_latest/EnergyTSF_20/TPGNN/datasets/W_Germany_processed_0_all.csv"

# 网格搜索参数
for lr in 0.001 0.0005
do
  for bs in 16 32
  do
    for r in 0.1 0.5
    do
      export WANDB_NAME="PeMS_lr${lr}_bs${bs}_r${r}"

      echo "Running config: lr=${lr}, bs=${bs}, r=${r}"

      python /mnt/webscistorage/cc7738/ws_joella/EnergyTSF_latest/EnergyTSF_20/TPGNN/TPGNN/main_tpgnn.py train \
        --device=0 \
        --n_day=4 \
        --n_route=12 \
        --n_his=12 \
        --n_pred=12 \
        --n_train=7 \
        --n_val=2 \
        --n_test=1 \
        --mode=1 \
        --name="easy_lr${lr}_bs${bs}_r${r}" \
        --data_path="${DATA_PATH}" \
        --adj_matrix_path="${ADJ_PATH}" \
        --seq_in_len=12 \
        --seq_out_len=12 \
        --stamp_path="${STAMP_PATH}" \
        --epochs=50 \
        --lr=$lr \
        --batch_size=$bs \
        --r=$r

    done
  done
done


