#!/bin/bash
set -e

# ========= 选择要跑的实验（0=全部；1=Baseline；2=幂律；3=MixPropDual；4=Chebyshev；5=无对角 6=powermixhop）=========
EXP_ID=${EXP_ID:-0}

# ========= 基本 / 训练设置 =========
DEVICE=${DEVICE:-cuda:0}
EPOCHS=${EPOCHS:-5}
ADJTYPE=${ADJTYPE:-doubletransition}
PRINT_EVERY=${PRINT_EVERY:-50}

# ========= wandb 设置 =========
export WANDB_PROJECT=${WANDB_PROJECT:-GraphWaveNet}
export WANDB_ENTITY=${WANDB_ENTITY:-}
export WANDB_MODE=${WANDB_MODE:-online}       # online/offline
export WANDB_DIR=${WANDB_DIR:-./wandb_runs}
mkdir -p "$WANDB_DIR"

# ========= 结果表（CSV）路径 =========
export RESULTS_CSV=${RESULTS_CSV:-./results.csv}

# ========= 环境开关（依然可用于 ablation，但不再画邻接图）=========
export GWN_DIAG_MODE=${GWN_DIAG_MODE:-self_and_neighbor}  # neighbor/self_and_neighbor

# ========= 网格（可用环境变量覆盖）=========
# 支持对 DATA 与 BATCH 做网格；--未显式给 DATA_LIST/BATCH_LIST，--退到 DATA/BATCH，--回退到默认
DATA_LIST=(${DATA_LIST:-${DATA:-data/FRANCE}})
BATCH_LIST=(${BATCH_LIST:-${BATCH:-64}})

SEQ_LIST=(${SEQ_LIST:-12})                # 3 6 12 24
PRED_LIST=(${PRED_LIST:-12})              # 3 6 12 24
LR_LIST=(${LR_LIST:-0.001 0.0001 0.00001})
DROPOUT_LIST=(${DROPOUT_LIST:-0.3})
NHID_LIST=(${NHID_LIST:-64})
WD_LIST=(${WD_LIST:-0.0001})


# —— 空间层数消融：默认固定 blocks=4，layers 可在 [2,1] 间切换 —— 
BLOCKS_LIST=(${BLOCKS_LIST:-${BLOCKS:-4}}) ##############
LAYERS_LIST=(${LAYERS_LIST:-${LAYERS:-2}}) ################




# ========= 跑一个实验（不再调用 _viz_probe.py）=========


run_one () {
  local EXP_GROUP="$1"
  export MODEL_NAME="$EXP_GROUP"
  local SEQ="$2"; local PRED="$3"; local LR="$4"; local DROPOUT="$5"; local NHID="$6"; local WD="$7"

  # blocks/layers（默认 4×2）
  local BLOCKS="${BLOCKS:-4}"
  local LAYERS="${LAYERS:-2}"

  local EXP_NAME="${EXP_GROUP}_data$(basename "$DATA")_bs${BATCH}_seq${SEQ}_pred${PRED}_lr${LR}_do${DROPOUT}_hid${NHID}_wd${WD}_b${BLOCKS}_l${LAYERS}"

  # 可选 addaptadj 开关：DISABLE_ADAPTADJ=1 时不加；否则加
  local ADDAPT=
  if [[ "${DISABLE_ADAPTADJ:-0}" != "1" ]]; then
    ADDAPT="--addaptadj"
  fi

  # wandb config
  local CFG_JSON
  CFG_JSON=$(cat <<JSON
{"exp_group":"$EXP_GROUP",
 "data":"${DATA:-data/FRANCE}","device":"${DEVICE:-cuda:0}","epochs":${EPOCHS:-5},"batch_size":${BATCH:-64},
 "seq_length":${SEQ:-12},"pred_length":${PRED:-12},"learning_rate":${LR:-0.001},"dropout":${DROPOUT:-0.3},
 "nhid":${NHID:-64},"weight_decay":${WD:-0.0001},"adjtype":"${ADJTYPE:-doubletransition}",
 "gcn_bool":true,"addaptadj":$([[ "${DISABLE_ADAPTADJ:-0}" == "1" ]] && echo false || echo true),
 "randomadj":true,"print_every":${PRINT_EVERY:-50},
 "blocks":${BLOCKS},"layers":${LAYERS}}
JSON
)

  # 用“数组”构建命令，避免空参数
  local CMD=(python train.py
    --data "$DATA" --device "$DEVICE" --batch_size "$BATCH" --epochs "$EPOCHS"
    --seq_length "$SEQ" --pred_length "$PRED"
    --learning_rate "$LR" --dropout "$DROPOUT" --nhid "$NHID"
    --weight_decay "$WD" --print_every "$PRINT_EVERY"
    --gcn_bool --randomadj --adjtype "$ADJTYPE"
    --blocks "$BLOCKS" --layers "$LAYERS"
  )
  # 按需追加 --addaptadj
  if [[ -n "$ADDAPT" ]]; then
    CMD+=("$ADDAPT")
  fi

  echo ">>> [$EXP_NAME]"
  echo "[CMD] ${CMD[*]}"
  python _wandb_proxy.py --project "$WANDB_PROJECT" --name "$EXP_NAME" --config "$CFG_JSON" --cmd "${CMD[*]}"
}


# ======================== 实验开关（按 EXP_ID 选择） ========================

# ---- 实验 1：Baseline ----
if [[ $EXP_ID -eq 0 || $EXP_ID -eq 1 ]]; then
  echo "==> EXP 1: Baseline"
  export GWN_USE_POWER=0
  export GWN_USE_CHEBY=0
  export GWN_DIAG_MODE=self_and_neighbor
  for DATA in "${DATA_LIST[@]}"; do
    for BATCH in "${BATCH_LIST[@]}"; do
      # Baseline 
      for ((i=0; i<${#SEQ_LIST[@]}; i++)); do
        SEQ=${SEQ_LIST[$i]}
        PRED=${PRED_LIST[$i]}
        for LR in "${LR_LIST[@]}"; do
          for DROPOUT in "${DROPOUT_LIST[@]}"; do
            for NHID in "${NHID_LIST[@]}"; do
              for WD in "${WD_LIST[@]}"; do
                run_one "Baseline" "$SEQ" "$PRED" "$LR" "$DROPOUT" "$NHID" "$WD"
              done
            done
          done
        done
      done
    done
  done
fi

# ---- 实验 2：幂律传播 ----

# if [[ $EXP_ID -eq 0 || $EXP_ID -eq 2 ]]; then
#   echo "==> EXP 2: PowerLaw"
#   export GWN_USE_POWER=1; export GWN_USE_CHEBY=0; export GWN_DIAG_MODE=self_and_neighbor
#   for DATA in "${DATA_LIST[@]}"; do for BATCH in "${BATCH_LIST[@]}"; do
#     for SEQ in "${SEQ_LIST[@]}"; do for PRED in "${PRED_LIST[@]}"; do
#       for LR in "${LR_LIST[@]}"; do for DROPOUT in "${DROPOUT_LIST[@]}"; do
#         for NHID in "${NHID_LIST[@]}"; do for WD in "${WD_LIST[@]}"; do
#           run_one "PowerLaw" "$SEQ" "$PRED" "$LR" "$DROPOUT" "$NHID" "$WD"
#         done; done
#       done; done
#     done; done
#   done; done
# fi

# ---- 实验 2：PowerLaw ----
if [[ $EXP_ID -eq 0 || $EXP_ID -eq 2 ]]; then
  echo "==> EXP 2: PowerLaw Ablation"

  # Power Law 专用开关
  export GWN_USE_POWER=1
  export GWN_USE_CHEBY=0
  export GWN_USE_MIXPROP=0

  # ====== 实验网格 ======
  # 阶数 2 3 4
  ORDER_LIST=(2 3)

  # 幂律系数初始化策略（根据这个开关调整实现）
  # plain = [1,1,...]  decay = [1,0.5,0.25...]  softmax = softmax归一化
  COEF_INIT_LIST=("plain" "decay" "softmax")

  # 学习率
  LR_LIST=(0.001 0.0005 0.0001)

  # Dropout
  DROPOUT_LIST=(0.3 0.5)

  # Diag mode
  DIAG_LIST=("self_and_neighbor" "neighbor")

  for DATA in "${DATA_LIST[@]}"; do
    for BATCH in "${BATCH_LIST[@]}"; do
      for SEQ in "${SEQ_LIST[@]}"; do
        for PRED in "${PRED_LIST[@]}"; do
          for ORDER in "${ORDER_LIST[@]}"; do
            for INIT in "${COEF_INIT_LIST[@]}"; do
              for LR in "${LR_LIST[@]}"; do
                for DROPOUT in "${DROPOUT_LIST[@]}"; do
                  for DIAG in "${DIAG_LIST[@]}"; do
                    for NHID in "${NHID_LIST[@]}"; do
                      for WD in "${WD_LIST[@]}"; do

                        export GWN_POWER_ORDER=$ORDER
                        export GWN_POWER_INIT=$INIT
                        export GWN_DIAG_MODE=$DIAG

                        run_one "PowerLaw_o${ORDER}_${INIT}_${DIAG}" \
                                "$SEQ" "$PRED" "$LR" "$DROPOUT" "$NHID" "$WD"

                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
fi

# ---- 实验 3：MixPropDual ----
if [[ $EXP_ID -eq 0 || $EXP_ID -eq 3 ]]; then
  echo "==> EXP 3: MixPropDual"
  export GWN_USE_MIXPROP=1
  export GWN_MIXPROP_K=${GWN_MIXPROP_K:-3}
  export GWN_ADJ_DROPOUT=${GWN_ADJ_DROPOUT:-0.1}
  export GWN_ADJ_TEMP=${GWN_ADJ_TEMP:-1.0}
  export GWN_USE_POWER=0; export GWN_USE_CHEBY=0; export GWN_DIAG_MODE=self_and_neighbor
  for DATA in "${DATA_LIST[@]}"; do for BATCH in "${BATCH_LIST[@]}"; do
    for SEQ in "${SEQ_LIST[@]}"; do for PRED in "${PRED_LIST[@]}"; do
      for LR in "${LR_LIST[@]}"; do for DROPOUT in "${DROPOUT_LIST[@]}"; do
        for NHID in "${NHID_LIST[@]}"; do for WD in "${WD_LIST[@]}"; do
          run_one "MixPropDual" "$SEQ" "$PRED" "$LR" "$DROPOUT" "$NHID" "$WD"
        done; done
      done; done
    done; done
  done; done
fi

# ---- 实验 4：Chebyshev ----
if [[ $EXP_ID -eq 0 || $EXP_ID -eq 4 ]]; then
  echo "==> EXP 4: Chebyshev"
  export GWN_USE_POWER=0; export GWN_USE_CHEBY=1
  export GWN_CHEBY_K=${GWN_CHEBY_K:-3}
  export GWN_DIAG_MODE=self_and_neighbor
  for DATA in "${DATA_LIST[@]}"; do for BATCH in "${BATCH_LIST[@]}"; do
    for SEQ in "${SEQ_LIST[@]}"; do for PRED in "${PRED_LIST[@]}"; do
      for LR in "${LR_LIST[@]}"; do for DROPOUT in "${DROPOUT_LIST[@]}"; do
        for NHID in "${NHID_LIST[@]}"; do for WD in "${WD_LIST[@]}"; do
          run_one "Chebyshev" "$SEQ" "$PRED" "$LR" "$DROPOUT" "$NHID" "$WD"
        done; done
      done; done
    done; done
  done; done
fi

# ---- 实验 5：无对角邻接 ----
if [[ $EXP_ID -eq 0 || $EXP_ID -eq 5 ]]; then
  echo "==> EXP 5: NoDiagonal"
  export GWN_USE_POWER=0; export GWN_USE_CHEBY=0; export GWN_DIAG_MODE=neighbor
  for DATA in "${DATA_LIST[@]}"; do for BATCH in "${BATCH_LIST[@]}"; do
    for SEQ in "${SEQ_LIST[@]}"; do for PRED in "${PRED_LIST[@]}"; do
      for LR in "${LR_LIST[@]}"; do for DROPOUT in "${DROPOUT_LIST[@]}"; do
        for NHID in "${NHID_LIST[@]}"; do for WD in "${WD_LIST[@]}"; do
          run_one "NoDiagonal" "$SEQ" "$PRED" "$LR" "$DROPOUT" "$NHID" "$WD"
        done; done
      done; done
    done; done
  done; done
fi


# ---- 实验 6：PowerMixDual ----
if [[ $EXP_ID -eq 0 || $EXP_ID -eq 6 ]]; then
  echo "==> EXP 6: PowerMixDual"
  export GWN_USE_POWERMIX=1
  export GWN_USE_POWER=0; export GWN_USE_CHEBY=0; export GWN_USE_MIXPROP=0
  export GWN_DIAG_MODE=self_and_neighbor

  # ====== 网格 ======
  ORDER_LIST=(1 2 3)                          # 幂律最大阶数
  COEF_INIT_LIST=("plain" "decay" "softmax") # 幂律初始化
  K_LIST=(1 2 3)                              # MixPropDual 递推步长
  DROPOUT_LIST=(0.3 0.5)                    # A-dropout
  TEMP_LIST=(1.0 0.5)                       # 温度
  DIAG_LIST=("self_and_neighbor" "neighbor")
  LR_LIST=(0.001 0.0001)

  for DATA in "${DATA_LIST[@]}"; do
    for BATCH in "${BATCH_LIST[@]}"; do
      for SEQ in "${SEQ_LIST[@]}"; do
        for PRED in "${PRED_LIST[@]}"; do
          for ORDER in "${ORDER_LIST[@]}"; do
            for INIT in "${COEF_INIT_LIST[@]}"; do
              for K in "${K_LIST[@]}"; do
                for DROPOUT in "${DROPOUT_LIST[@]}"; do
                  for TEMP in "${TEMP_LIST[@]}"; do
                    for DIAG in "${DIAG_LIST[@]}"; do
                      for LR in "${LR_LIST[@]}"; do
                        for NHID in "${NHID_LIST[@]}"; do
                          for WD in "${WD_LIST[@]}"; do

                            export GWN_POWER_ORDER=$ORDER
                            export GWN_POWER_INIT=$INIT
                            export GWN_POWERMIX_K=$K
                            export GWN_POWERMIX_DROPOUT=$DROPOUT
                            export GWN_POWERMIX_TEMP=$TEMP
                            export GWN_DIAG_MODE=$DIAG

                            run_one "PowerMixDual_o${ORDER}_${INIT}_K${K}_${DIAG}" \
                                    "$SEQ" "$PRED" "$LR" "$DROPOUT" "$NHID" "$WD"

                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
fi



# ---- 实验 7：PowerMixDual - 空间层数消融（只改 layers；blocks 固定）----
if [[ $EXP_ID -eq 0 || $EXP_ID -eq 7 ]]; then
  echo "==> EXP 7: PowerMixDual Spatial-Layers Ablation"
  export GWN_USE_POWERMIX=1
  export GWN_USE_POWER=0; export GWN_USE_CHEBY=0; export GWN_USE_MIXPROP=0
  export GWN_DIAG_MODE=${GWN_DIAG_MODE:-self_and_neighbor}

  # 其余保持与 EXP 6 一致，避免口径不一致
  ORDER_LIST=(2 3)
  COEF_INIT_LIST=("plain" "decay" "softmax")
  K_LIST=(2 3)
  DROPOUT_LIST=(0.3)          # 消融主轴尽量少改项，可按需扩展
  TEMP_LIST=(1.0)
  DIAG_LIST=("${GWN_DIAG_MODE}")
  LR_LIST=(0.001 0.0001)

  for DATA in "${DATA_LIST[@]}"; do
    for BATCH in "${BATCH_LIST[@]}"; do
      for SEQ in "${SEQ_LIST[@]}"; do
        for PRED in "${PRED_LIST[@]}"; do
          for BLOCKS in "${BLOCKS_LIST[@]}"; do             # 通常 = 4
            for LAYERS in "${LAYERS_LIST[@]}"; do           # 这里做 2 → 1
              for ORDER in "${ORDER_LIST[@]}"; do
                for INIT in "${COEF_INIT_LIST[@]}"; do
                  for K in "${K_LIST[@]}"; do
                    for DROPOUT in "${DROPOUT_LIST[@]}"; do
                      for TEMP in "${TEMP_LIST[@]}"; do
                        for DIAG in "${DIAG_LIST[@]}"; do
                          for LR in "${LR_LIST[@]}"; do
                            for NHID in "${NHID_LIST[@]}"; do
                              for WD in "${WD_LIST[@]}"; do

                                export GWN_POWER_ORDER=$ORDER
                                export GWN_POWER_INIT=$INIT
                                export GWN_POWERMIX_K=$K
                                export GWN_POWERMIX_DROPOUT=$DROPOUT
                                export GWN_POWERMIX_TEMP=$TEMP
                                export GWN_DIAG_MODE=$DIAG

                                # 重要：设置 EXP_GROUP 易于区分
                                run_one "PMD_layersAblation_b${BLOCKS}_l${LAYERS}_o${ORDER}_${INIT}_K${K}_${DIAG}" \
                                        "$SEQ" "$PRED" "$LR" "$DROPOUT" "$NHID" "$WD"

                              done
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
fi



# ---- 实验 8：Graph Variants for PowerMixDual（Random / No-Temporal / Single-Graph）----
if [[ $EXP_ID -eq 0 || $EXP_ID -eq 8 ]]; then
  echo "==> EXP 8: PowerMixDual Graph Variants"
  export GWN_USE_POWERMIX=1
  export GWN_USE_POWER=0; export GWN_USE_CHEBY=0; export GWN_USE_MIXPROP=0
  export GWN_DIAG_MODE=${GWN_DIAG_MODE:-self_and_neighbor}

  # 固定一组稳妥设置（你也可改成你论文里的best组合）
  ORDER_LIST=(1 2 3)               # 幂律阶
  COEF_INIT_LIST=("softmax")     # 系数初始化
  K_LIST=(2)                     # 递推步
  DROPOUT_LIST=(0.3)             # A-dropout
  TEMP_LIST=(1.0)
  DIAG_LIST=("${GWN_DIAG_MODE}")
  LR_LIST=(0.001)

  # 三种图变体
  VARIANTS=("random" "no_temporal" "single_graph")

  for DATA in "${DATA_LIST[@]}"; do
    for BATCH in "${BATCH_LIST[@]}"; do
      for SEQ in "${SEQ_LIST[@]}"; do
        for PRED in "${PRED_LIST[@]}"; do
          for ORDER in "${ORDER_LIST[@]}"; do
            for INIT in "${COEF_INIT_LIST[@]}"; do
              for K in "${K_LIST[@]}"; do
                for DROPOUT in "${DROPOUT_LIST[@]}"; do
                  for TEMP in "${TEMP_LIST[@]}"; do
                    for DIAG in "${DIAG_LIST[@]}"; do
                      for LR in "${LR_LIST[@]}"; do
                        for NHID in "${NHID_LIST[@]}"; do
                          for WD in "${WD_LIST[@]}"; do
                            for V in "${VARIANTS[@]}"; do

                              export GWN_POWER_ORDER=$ORDER
                              export GWN_POWER_INIT=$INIT
                              export GWN_POWERMIX_K=$K
                              export GWN_POWERMIX_DROPOUT=$DROPOUT
                              export GWN_POWERMIX_TEMP=$TEMP
                              export GWN_DIAG_MODE=$DIAG

                              # 缺省：启用自适应邻接
                              DISABLE_ADAPTADJ=0
                              unset GWN_RANDOM_BASE_GRAPH
                              unset GWN_SECOND_GRAPH_FIXED
                              unset GWN_DISABLE_SECOND_GRAPH

                              if [[ "$V" == "random" ]]; then
                                export GWN_RANDOM_BASE_GRAPH=1
                                EXP_TAG="PMD_randBase"
                              elif [[ "$V" == "no_temporal" ]]; then
                                export GWN_SECOND_GRAPH_FIXED=1   # 第二路改用固定基础图
                                DISABLE_ADAPTADJ=1                # 同时关掉 Graph WaveNet 外层的自适应支持（可选）
                                EXP_TAG="PMD_noTemporal"
                              elif [[ "$V" == "single_graph" ]]; then
                                export GWN_DISABLE_SECOND_GRAPH=1 # 只保留第一路
                                DISABLE_ADAPTADJ=1
                                EXP_TAG="PMD_singleGraph"
                              fi

                              # 让 run_one 能根据 DISABLE_ADAPTADJ 决定是否传 --addaptadj
                              export DISABLE_ADAPTADJ=$DISABLE_ADAPTADJ

                              run_one "${EXP_TAG}_o${ORDER}_${INIT}_K${K}_${DIAG}" \
                                      "$SEQ" "$PRED" "$LR" "$DROPOUT" "$NHID" "$WD"

                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
fi



echo "✅ 实验完成（EXP_ID=$EXP_ID）。wandb 项目：$WANDB_PROJECT"
echo "📄 结果已累计写入 CSV：$RESULTS_CSV" 


# git: /mnt/webscistorage/cc7738/ws_joella/CorrGCN
# cd /mnt/webscistorage/cc7738/ws_joella/CorrGCN/Graph-DynGCN/


# RESULTS_CSV=./results_exchange_graph_variants.csv EXP_ID=8 DATA_LIST="data/EXCHANGE" SEQ_LIST="12" PRED_LIST="1 3 6 12" BLOCKS_LIST="4" LAYERS_LIST="2" DEVICE=cuda:0 bash run_experiments_ab.sh
# RESULTS_CSV=./results_exchange_layers.csv EXP_ID=7 DATA_LIST="data/EXCHANGE" SEQ_LIST="12" PRED_LIST="1 3 6 12" BLOCKS_LIST="4" LAYERS_LIST="2 1" bash run_experiments_ab.sh
# EXP_ID=7 DATA_LIST="data/EXCHANGE" SEQ_LIST="1 3 6 12" PRED_LIST="1 3 6 12" BLOCKS_LIST="4" LAYERS_LIST="2 1" bash run_experiments_ab.sh
# WANDB_PROJECT=PowerMixAblation WANDB_ENTITY=gabiyueyue26 WANDB_MODE=online WANDB_DIR=./wandb_runs/EXCHANGE_layers RESULTS_CSV=./results_exchange_layers.csv EXP_ID=7 DATA_LIST="data/EXCHANGE" SEQ_LIST="12" PRED_LIST="12" BLOCKS_LIST="4" LAYERS_LIST="2 1" DEVICE=cuda:0 bash run_experiments_ab.sh
# WANDB_PROJECT=ExAblGraph WANDB_ENTITY=gabiyueyue26 WANDB_MODE=online WANDB_DIR=./wandb_runs/EXCHANGE_graph RESULTS_CSV=./results_exchange_graph_variants.csv DATA_LIST="data/EXCHANGE" SEQ_LIST="1 3 6 12" PRED_LIST="1 3 6 12" DEVICE=cuda:0 GWN_RANDOM_BASE_GRAPH=1 EXP_ID=6 bash run_experiments_ab.sh && WANDB_PROJECT=ExAblGraph WANDB_ENTITY=gabiyueyue26 WANDB_MODE=online WANDB_DIR=./wandb_runs/EXCHANGE_graph RESULTS_CSV=./results_exchange_graph_variants.csv DATA_LIST="data/EXCHANGE" SEQ_LIST="1 3 6 12" PRED_LIST="1 3 6 12" DEVICE=cuda:0 GWN_SECOND_GRAPH_FIXED=1 EXP_ID=6 bash run_experiments_ab.sh && WANDB_PROJECT=ExAblGraph WANDB_ENTITY=gabiyueyue26 WANDB_MODE=online WANDB_DIR=./wandb_runs/EXCHANGE_graph RESULTS_CSV=./results_exchange_graph_variants.csv DATA_LIST="data/EXCHANGE" SEQ_LIST="1 3 6 12" PRED_LIST="1 3 6 12" DEVICE=cuda:0 GWN_DISABLE_SECOND_GRAPH=1 EXP_ID=6 bash run_experiments_ab.sh


### 5 MODEL ###
# WANDB_PROJECT=GraphWaveNet-PowerLaw_1 RESULTS_CSV=./results_PowerLaw_1.csv DATA_LIST="data/SYNTHETIC_EASY data/SYNTHETIC_MEDIUM data/SYNTHETIC_HARD data/SYNTHETIC_VERY_HARD" BATCH_LIST="32 64 128" LR_LIST="0.001 0.0001 0.00001" EXP_ID=2 bash run_experiments_1.sh
# WANDB_PROJECT=GraphWaveNet-Baseline_1 RESULTS_CSV=./results_Baseline_1_1.csv DATA_LIST="data/SYNTHETIC_EASY data/SYNTHETIC_MEDIUM data/SYNTHETIC_HARD data/SYNTHETIC_VERY_HARD" BATCH_LIST="32 64 128" LR_LIST="0.001 0.0001 0.00001" EXP_ID=1 bash run_experiments_1.sh
# WANDB_PROJECT=GraphWaveNet-MixPropDual_1 RESULTS_CSV=./results_MixPropDual_1.csv DATA_LIST="data/SYNTHETIC_EASY data/SYNTHETIC_MEDIUM data/SYNTHETIC_HARD data/SYNTHETIC_VERY_HARD" BATCH_LIST="32 64 128" LR_LIST="0.001 0.0001 0.00001" EXP_ID=3 bash run_experiments_1.sh
# WANDB_PROJECT=GraphWaveNet-Chebyshev_1 RESULTS_CSV=./results_Chebyshev_1.csv DATA_LIST="data/SYNTHETIC_EASY data/SYNTHETIC_MEDIUM data/SYNTHETIC_HARD data/SYNTHETIC_VERY_HARD" BATCH_LIST="32 64 128" LR_LIST="0.001 0.0001 0.00001" EXP_ID=4 bash run_experiments_1.sh
# WANDB_PROJECT=GraphWaveNet-NoDiagonal_1 RESULTS_CSV=./results_NoDiagonal_1.csv DATA_LIST="data/SYNTHETIC_EASY data/SYNTHETIC_MEDIUM data/SYNTHETIC_HARD data/SYNTHETIC_VERY_HARD" BATCH_LIST="32 64 128" LR_LIST="0.001 0.0001 0.00001" EXP_ID=5 bash run_experiments_1.sh


# WANDB_PROJECT=GraphWaveNet-PowerMixDual RESULTS_CSV=./results_PowerMixDual.csv DATA_LIST="data/SYNTHETIC_EASY data/SYNTHETIC_MEDIUM data/SYNTHETIC_HARD data/SYNTHETIC_VERY_HARD" BATCH_LIST="32 64 128" LR_LIST="0.001 0.0001 0.00001" EXP_ID=6 bash run_experiments_1.sh
# WANDB_MODE=disabled WANDB_PROJECT=GraphWaveNet-PowerMixDual RESULTS_CSV=./results_PowerMixDual.csv \DATA_LIST="data/SYNTHETIC_EASY data/SYNTHETIC_MEDIUM data/SYNTHETIC_HARD data/SYNTHETIC_VERY_HARD" \BATCH_LIST="32 64 128" LR_LIST="0.001 0.0001 0.00001" EXP_ID=6 bash run_experiments_1.sh

# EPOCHS=5 WANDB_MODE=disabled WANDB_PROJECT=GraphWaveNet-PowerMixDual RESULTS_CSV=./results_PowerMixDual_Elec_OUR.csv DATA_LIST="data/ELECTRICITY" BATCH_LIST="8" LR_LIST="0.001 0.0001 0.00001" EXP_ID=6 bash run_experiments_ab.sh
# EPOCHS=5 WANDB_MODE=disabled WANDB_PROJECT=GraphWaveNet-PowerMixDual RESULTS_CSV=./results_PowerMixDual_Elec_POWER.csv DATA_LIST="data/ELECTRICITY" BATCH_LIST="8" LR_LIST="0.001 0.0001 0.00001" EXP_ID=2 bash run_experiments_ab.sh
# EPOCHS=5 WANDB_MODE=disabled WANDB_PROJECT=GraphWaveNet-PowerMixDual RESULTS_CSV=./results_PowerMixDual_Elec_DUAL.csv DATA_LIST="data/ELECTRICITY" BATCH_LIST="8" LR_LIST="0.001 0.0001 0.00001" EXP_ID=3 bash run_experiments_ab.sh
# EPOCHS=5 WANDB_MODE=disabled WANDB_PROJECT=GraphWaveNet-PowerMixDual RESULTS_CSV=./results_PowerMixDual_Solar.csv DATA_LIST="data/SOLAR" BATCH_LIST="16" LR_LIST="0.001 0.0001 0.00001" EXP_ID=6 bash run_experiments_ab.sh
# EPOCHS=20 WANDB_MODE=disabled WANDB_PROJECT=GraphWaveNet-PowerMixDual RESULTS_CSV=./results_PowerMixDual_EXCHANGE_OUR.csv DATA_LIST="data/EXCHANGE" BATCH_LIST="64 128 256" LR_LIST="0.001 0.0001 0.00001" EXP_ID=6 bash run_experiments_ab.sh
# EPOCHS=20 WANDB_MODE=disabled WANDB_PROJECT=GraphWaveNet-PowerMixDual RESULTS_CSV=./results_PowerMixDual_EXCHANGE_POWER.csv DATA_LIST="data/EXCHANGE" BATCH_LIST="64 128 256" LR_LIST="0.001 0.0001 0.00001" EXP_ID=2 bash run_experiments_ab.sh
# EPOCHS=20 WANDB_MODE=disabled WANDB_PROJECT=GraphWaveNet-PowerMixDual RESULTS_CSV=./results_PowerMixDual_EXCHANGE_DUAL.csv DATA_LIST="data/EXCHANGE" BATCH_LIST="64 128 256" LR_LIST="0.001 0.0001 0.00001" EXP_ID=3 bash run_experiments_ab.sh


### DATA_LIST="data/SYNTHETIC_EASY data/SYNTHETIC_MEDIUM data/SYNTHETIC_HARD data/SYNTHETIC_VERY_HARD" BATCH_LIST="32 64 128" LR_LIST="0.001 0.0001 0.00001" EXP_ID=5 bash run_experiments.sh

# WANDB_PROJECT=GraphWaveNet-Baseline_1 RESULTS_CSV=./results_Baseline_1_1.csv DATA_LIST="data/FRANCE data/SYNTHETIC_MEDIUM data/SYNTHETIC_HARD data/SYNTHETIC_VERY_HARD" BATCH_LIST="32 64 128" LR_LIST="0.001 0.0001 0.00001" EXP_ID=1 bash run_experiments_1.sh

# WANDB_PROJECT=GraphWaveNet-PowerLaw-2 RESULTS_CSV=./results_PowerLaw-3.csv DATA_LIST="data/SYNTHETIC_HARD" BATCH_LIST="32 64 128" LR_LIST="0.001 0.0001 0.00001" EXP_ID=2 bash run_experiments.sh
# WANDB_PROJECT=GraphWaveNet-PowerLaw-2 RESULTS_CSV=./results_PowerLaw-4.csv DATA_LIST="data/SYNTHETIC_VERY_HARD" BATCH_LIST="32 64 128" LR_LIST="0.001 0.0001 0.00001" EXP_ID=2 bash run_experiments.sh


# srun -p 4090 --nodelist=aifb-websci-gpunode1 --gres=gpu:1 -t 4:00:00 --pty bash -i
# srun -p 4090 --nodelist=aifb-websci-gpunode1 --gres=gpu:2 -t 0:30:00 --pty bash -i
# srun -p 4090 --nodelist=aifb-websci-gpunode1 --gres=gpu:4 -t 1:00:00 --pty bash -i

### nvidia-smi
### squeue -u $USER
### conda activate Energy-TSF
### cd /mnt/webscistorage/cc7738/ws_joella/EnergyTSF
# cd /mnt/webscistorage/cc7738/ws_joella/EnergyTSF/GNN/Graph-WaveNet-master-origin
# cd /mnt/webscistorage/cc7738/ws_chen/GraphDynamic
# wandb login

