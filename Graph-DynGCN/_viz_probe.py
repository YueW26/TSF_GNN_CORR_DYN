#!/bin/bash
set -e

# ========= 选择要跑的实验（0=全部；1=Baseline；2=幂律；3=MixPropDual；4=Chebyshev；5=无对角）=========
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
# 支持对 DATA 与 BATCH 做网格；若未显式给 DATA_LIST/BATCH_LIST，则回退到 DATA/BATCH，再回退到默认
DATA_LIST=(${DATA_LIST:-${DATA:-data/FRANCE}})
BATCH_LIST=(${BATCH_LIST:-${BATCH:-64}})

SEQ_LIST=(${SEQ_LIST:-12})                # 也可：3 6 12 24
PRED_LIST=(${PRED_LIST:-12})              # 也可：3 6 12 24
LR_LIST=(${LR_LIST:-0.001 0.0001 0.00001})
DROPOUT_LIST=(${DROPOUT_LIST:-0.3})
NHID_LIST=(${NHID_LIST:-64})
WD_LIST=(${WD_LIST:-0.0001})

# ========= wandb 代理：解析日志 + 写入 CSV =========
cat > _wandb_proxy.py <<'PY'
import os, re, sys, json, shlex, subprocess, csv, time
import wandb

def ensure_csv_header(csv_path, fieldnames):
    exists = os.path.exists(csv_path)
    # 创建目录（若传入了带路径的文件）
    os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None
    if not exists:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()

def append_csv(csv_path, row, fieldnames):
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow(row)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", required=True, help="json string")
    ap.add_argument("--cmd", required=True, help="command to run")
    args = ap.parse_args()

    # 解析配置
    cfg = json.loads(args.config)

    # wandb 初始化
    run = wandb.init(project=args.project, name=args.name, config=cfg, dir=os.environ.get("WANDB_DIR", "./wandb_runs"))
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("valid/*", step_metric="epoch")
    wandb.define_metric("test/*",  step_metric="epoch")

    # 记录环境开关
    gwn_flags = {k:v for k,v in os.environ.items() if k.startswith("GWN_")}
    wandb.config.update({"env_flags": gwn_flags}, allow_val_change=True)

    # 启动训练子进程
    proc = subprocess.Popen(
        shlex.split(args.cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True
    )

    # 日志正则
    re_iter   = re.compile(r"Iter:\s*(\d+),\s*Train Loss:\s*([\d\.eE+-]+),\s*Train MAPE:\s*([\d\.eE+-]+),\s*Train RMSE:\s*([\d\.eE+-]+)")
    re_epoch  = re.compile(r"Epoch:\s*(\d+),.*Valid Loss:\s*([\d\.eE+-]+),\s*Valid MAPE:\s*([\d\.eE+-]+),\s*Valid RMSE:\s*([\d\.eE+-]+)")
    re_epoch_time = re.compile(r"Epoch:\s*(\d+),\s*Inference Time:")
    re_best   = re.compile(r"The valid loss on best model is\s*([\d\.eE+-]+)")
    re_hz     = re.compile(r"Evaluate best model on test data for horizon\s*(\d+),\s*Test MAE:\s*([\d\.eE+-]+),\s*Test MAPE:\s*([\d\.eE+-]+),\s*Test RMSE:\s*([\d\.eE+-]+)")
    re_avg    = re.compile(r"On average over\s*(\d+)\s*horizons,\s*Test MAE:\s*([\d\.eE+-]+),\s*Test MAPE:\s*([\d\.eE+-]+),\s*Test RMSE:\s*([\d\.eE+-]+)")

    # 将在 CSV 里用到的字段
    csv_fields = [
        "exp_group","dataset","batch_size","seq","pred","lr","dropout","nhid","wd",
        "avg_test_mae","avg_test_rmse","avg_test_mape","best_valid_loss","run_name","timestamp"
    ]
    results_csv = os.environ.get("RESULTS_CSV", "./results.csv")
    ensure_csv_header(results_csv, csv_fields)

    current_epoch = 0
    best_valid = None
    avg_mae = avg_rmse = avg_mape = None

    for line in proc.stdout:
        sys.stdout.write(line); sys.stdout.flush()

        m = re_epoch_time.search(line)
        if m:
            current_epoch = int(m.group(1))
            wandb.log({"epoch": current_epoch}, step=current_epoch)
            continue

        m = re_iter.search(line)
        if m:
            wandb.log({
                "epoch": current_epoch,
                "train/loss": float(m.group(2)),
                "train/mape": float(m.group(3)),
                "train/rmse": float(m.group(4))
            }, step=current_epoch)
            continue

        m = re_epoch.search(line)
        if m:
            current_epoch = int(m.group(1))
            wandb.log({
                "epoch": current_epoch,
                "valid/loss": float(m.group(2)),
                "valid/mape": float(m.group(3)),
                "valid/rmse": float(m.group(4))
            }, step=current_epoch)
            continue

        m = re_best.search(line)
        if m:
            best_valid = float(m.group(1))
            wandb.summary["best_valid/loss"] = best_valid
            continue

        m = re_hz.search(line)
        if m:
            hz = int(m.group(1))
            wandb.log({
                "epoch": current_epoch,
                "test/horizon": hz,
                "test/horizon_mae":  float(m.group(2)),
                "test/horizon_mape": float(m.group(3)),
                "test/horizon_rmse": float(m.group(4))
            }, step=current_epoch)
            continue

        m = re_avg.search(line)
        if m:
            wandb.summary["test/avg_horizons"] = int(m.group(1))
            avg_mae  = float(m.group(2))
            avg_mape = float(m.group(3))
            avg_rmse = float(m.group(4))
            wandb.summary["test/avg_mae"]  = avg_mae
            wandb.summary["test/avg_mape"] = avg_mape
            wandb.summary["test/avg_rmse"] = avg_rmse
            continue

    proc.wait()

    # ---- 训练完成：把结果写进 CSV ----
    # 从 config 读取参数（带安全默认）
    cfg = dict(wandb.config)
    row = {
        "exp_group": cfg.get("exp_group",""),
        "dataset":   cfg.get("data",""),
        "batch_size":cfg.get("batch_size",""),
        "seq":       cfg.get("seq_length",""),
        "pred":      cfg.get("pred_length",""),
        "lr":        cfg.get("learning_rate",""),
        "dropout":   cfg.get("dropout",""),
        "nhid":      cfg.get("nhid",""),
        "wd":        cfg.get("weight_decay",""),
        "avg_test_mae":  avg_mae,
        "avg_test_rmse": avg_rmse,
        "avg_test_mape": avg_mape,
        "best_valid_loss": best_valid,
        "run_name":  args.name,
        "timestamp": int(time.time())
    }
    append_csv(results_csv, row, csv_fields)

    run.finish()
    sys.exit(proc.returncode)

if __name__ == "__main__":
    main()
PY

# ========= 跑一个实验（不再调用 _viz_probe.py）=========
run_one () {
  local EXP_GROUP="$1"   # Baseline / PowerLaw / MixPropDual / Chebyshev / NoDiagonal
  local SEQ="$2"; local PRED="$3"; local LR="$4"; local DROPOUT="$5"; local NHID="$6"; local WD="$7"
  local EXP_NAME="${EXP_GROUP}_data$(basename "$DATA")_bs${BATCH}_seq${SEQ}_pred${PRED}_lr${LR}_do${DROPOUT}_hid${NHID}_wd${WD}"

  # 训练配置（带默认兜底，避免空值导致 JSON 解析失败）
  local CFG_JSON
  CFG_JSON=$(cat <<JSON
{"exp_group":"$EXP_GROUP",
 "data":"${DATA:-data/FRANCE}","device":"${DEVICE:-cuda:0}","epochs":${EPOCHS:-5},"batch_size":${BATCH:-64},
 "seq_length":${SEQ:-12},"pred_length":${PRED:-12},"learning_rate":${LR:-0.001},"dropout":${DROPOUT:-0.3},
 "nhid":${NHID:-64},"weight_decay":${WD:-0.0001},"adjtype":"${ADJTYPE:-doubletransition}",
 "gcn_bool":true,"addaptadj":true,"randomadj":true,"print_every":${PRINT_EVERY:-50}}
JSON
)

  local CMD="python train.py \
    --data $DATA --device $DEVICE --batch_size $BATCH --epochs $EPOCHS \
    --seq_length $SEQ --pred_length $PRED \
    --learning_rate $LR --dropout $DROPOUT --nhid $NHID \
    --weight_decay $WD --print_every $PRINT_EVERY \
    --gcn_bool --addaptadj --randomadj --adjtype $ADJTYPE"

  echo ">>> [$EXP_NAME]"
  # echo "[CFG] $CFG_JSON"   # 若需排错可取消注释
  python _wandb_proxy.py --project "$WANDB_PROJECT" --name "$EXP_NAME" --config "$CFG_JSON" --cmd "$CMD"
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
      # Baseline 使用 SEQ/PRED 成对（与原脚本一致）
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
if [[ $EXP_ID -eq 0 || $EXP_ID -eq 2 ]]; then
  echo "==> EXP 2: PowerLaw"
  export GWN_USE_POWER=1; export GWN_USE_CHEBY=0; export GWN_DIAG_MODE=self_and_neighbor
  for DATA in "${DATA_LIST[@]}"; do for BATCH in "${BATCH_LIST[@]}"; do
    for SEQ in "${SEQ_LIST[@]}"; do for PRED in "${PRED_LIST[@]}"; do
      for LR in "${LR_LIST[@]}"; do for DROPOUT in "${DROPOUT_LIST[@]}"; do
        for NHID in "${NHID_LIST[@]}"; do for WD in "${WD_LIST[@]}"; do
          run_one "PowerLaw" "$SEQ" "$PRED" "$LR" "$DROPOUT" "$NHID" "$WD"
        done; done
      done; done
    done; done
  done; done
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

echo "✅ 实验完成（EXP_ID=$EXP_ID）。wandb 项目：$WANDB_PROJECT"
echo "📄 结果已累计写入 CSV：$RESULTS_CSV"