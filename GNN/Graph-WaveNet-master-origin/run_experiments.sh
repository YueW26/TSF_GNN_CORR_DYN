#!/bin/bash
set -e

# ========= 选择要跑的实验（0=全部；1=Baseline；2=幂律；3=MixPropDual；4=Chebyshev；5=无对角）=========
EXP_ID=${EXP_ID:-0}

# ========= 基本设置 =========
DATA=${DATA:-data/FRANCE}
DEVICE=${DEVICE:-cuda:0}
EPOCHS=${EPOCHS:-5}
BATCH=${BATCH:-64}
ADJTYPE=${ADJTYPE:-doubletransition}

# ========= wandb 项目信息 =========
export WANDB_PROJECT=${WANDB_PROJECT:-GraphWaveNet}
export WANDB_ENTITY=${WANDB_ENTITY:-}         # 
export WANDB_MODE=${WANDB_MODE:-online}       # online/offline
export WANDB_DIR=${WANDB_DIR:-./wandb_runs}   # 存放本地缓存
mkdir -p "$WANDB_DIR"

# ========= 可视化/探针开关 =========
export GWN_DIAG_MODE=${GWN_DIAG_MODE:-self_and_neighbor}  # neighbor/self_and_neighbor
LOG_TO_WANDB=${LOG_TO_WANDB:-1}     # 1=把图同步到 wandb；0=只本地保存
K_TOP=${K_TOP:-3}                    # Top-K 掩膜展示

# ========= 网格（改/用环境变量覆盖）=========
SEQ_LIST=(${SEQ_LIST:-12})
PRED_LIST=(${PRED_LIST:-12})
LR_LIST=(${LR_LIST:-0.001 0.0005})
DROPOUT_LIST=(${DROPOUT_LIST:-0.3 0.5})
NHID_LIST=(${NHID_LIST:-32 64})
WD_LIST=(${WD_LIST:-0.0001})
PRINT_EVERY=${PRINT_EVERY:-50}

# ========= 生成 wandb 代理脚本（统一用 epoch 作为横坐标）=========
cat > _wandb_proxy.py <<'PY'
import os, re, sys, json, shlex, subprocess
import wandb

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", required=True, help="json string")
    ap.add_argument("--cmd", required=True, help="command to run")
    args = ap.parse_args()

    cfg = json.loads(args.config)
    run = wandb.init(project=args.project, name=args.name, config=cfg, dir=os.environ.get("WANDB_DIR", "./wandb_runs"))

    # 统一定义：所有指标用 epoch 当 step
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("valid/*", step_metric="epoch")
    wandb.define_metric("test/*",  step_metric="epoch")

    # 记录环境开关
    gwn_flags = {k:v for k,v in os.environ.items() if k.startswith("GWN_")}
    wandb.config.update({"env_flags": gwn_flags}, allow_val_change=True)

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

    current_epoch = 0

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
            wandb.summary["best_valid/loss"] = float(m.group(1))
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
            wandb.summary["test/avg_mae"]  = float(m.group(2))
            wandb.summary["test/avg_mape"] = float(m.group(3))
            wandb.summary["test/avg_rmse"] = float(m.group(4))
            continue

    proc.wait()
    run.finish()
    sys.exit(proc.returncode)

if __name__ == "__main__":
    main()
PY

# ========= 生成可视化探针（离线读取 checkpoint & 数据做图）=========
cat > _viz_probe.py <<'PY'
import os, re, glob, argparse, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

def _natural_key(s): return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]
def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True); e = np.exp(x); return e / (e.sum(axis=axis, keepdims=True)+1e-12)

def load_base_adj(adjdata):
    import pickle
    with open(adjdata, 'rb') as f:
        obj = pickle.load(f, encoding='latin1')
    if isinstance(obj, (list, tuple)) and len(obj)==3:
        sensor_ids, sensor_id_to_ind, adj_mx = obj
    else:
        adj_mx = obj
    A = adj_mx[0] if isinstance(adj_mx, (list,tuple)) else adj_mx
    return np.asarray(A, dtype=np.float32)

def load_all_series(data_dir):
    def pick(x):  # x: [B,T,N,C] -> [BT,N] (C=0)
        B,T,N,C = x.shape
        return x.reshape(B*T, N, C)[:,:,0]
    xs=[]
    for sp in ["train","val","test"]:
        p=os.path.join(data_dir, f"{sp}.npz")
        if os.path.exists(p):
            d=np.load(p)
            xs.append(pick(d['x']))
    if not xs: raise FileNotFoundError(f"not found {data_dir}/(train|val|test).npz")
    return np.concatenate(xs, axis=0)  # [T,N]

def corr_matrix(X_TN):
    X = X_TN - X_TN.mean(0, keepdims=True)
    std = X.std(0, keepdims=True) + 1e-8
    Z = X / std
    return (Z.T @ Z) / (Z.shape[0]-1)

def find_checkpoints(save_dir):
    cks = glob.glob(os.path.join(save_dir, "*.pth"))
    epoch_ckpts, best_ckpt = [], None
    for p in cks:
        if "_epoch_" in os.path.basename(p): epoch_ckpts.append(p)
        if "_best_" in os.path.basename(p) and best_ckpt is None: best_ckpt = p
    epoch_ckpts.sort(key=_natural_key)
    return epoch_ckpts, best_ckpt

def extract_adaptive_A(sd):
    # 兼容 nodevec1/nodevec2/embed1/embed2
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    keys = list(sd.keys())
    def pick(*cands):
        for k in cands:
            if k in sd: return sd[k]
        return None
    E1 = pick('nodevec1','embed1','E1','module.nodevec1','graph.nodevec1')
    E2 = pick('nodevec2','embed2','E2','module.nodevec2','graph.nodevec2')
    if E1 is None and E2 is None:
        E = pick('nodevec','embed','E','module.nodevec')
        if E is None: return None
        E = E.detach().cpu().numpy()
        A = _softmax(np.maximum(E@E.T, 0.0), axis=1)
        return A.astype(np.float32)
    E1 = E1.detach().cpu().numpy()
    E2 = (E1 if E2 is None else E2.detach().cpu().numpy())
    A = _softmax(np.maximum(E1@E2.T, 0.0), axis=1)
    return A.astype(np.float32)

def plot_heatmap(M, title, out):
    plt.figure(figsize=(5,4)); plt.imshow(M, aspect='auto'); plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title); plt.tight_layout(); plt.savefig(out, dpi=180); plt.close()

def plot_series(xs, ys_dict, out, xlabel="epoch"):
    from matplotlib.ticker import MaxNLocator
    plt.figure(figsize=(6,4))
    for k,v in ys_dict.items(): plt.plot(xs, v, label=k, linewidth=2)
    plt.xlabel(xlabel); plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=180); plt.close()

def topk_mask(A, k=3):
    A2 = np.zeros_like(A); idx = np.argpartition(-A, kth=min(k-1, A.shape[1]-1), axis=1)[:, :k]
    rows = np.arange(A.shape[0])[:, None]; A2[rows, idx] = A[rows, idx]; return A2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--adjdata", required=True)
    ap.add_argument("--project", default="")
    ap.add_argument("--run_name", default="")
    ap.add_argument("--log_to_wandb", action="store_true")
    ap.add_argument("--k_top", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    figs_dir = os.path.join(args.save_dir, "viz"); os.makedirs(figs_dir, exist_ok=True)

    # Base adjacency & correlation
    A_base = load_base_adj(args.adjdata)
    plot_heatmap(A_base, "Base adjacency", os.path.join(figs_dir, "A_base.png"))

    X_TN = load_all_series(args.data)  # [T,N]
    C = corr_matrix(X_TN)
    plot_heatmap(C, "Correlation (Pearson)", os.path.join(figs_dir, "Corr_pearson.png"))

    # Epoch checkpoints
    epochs, diag_m, offdiag_m, asym_m, specr_m, simC_m = [], [], [], [], [], []
    epoch_ckpts, best_ckpt = find_checkpoints(args.save_dir)

    for pth in epoch_ckpts:
        m = re.search(r"_epoch_(\d+)_", os.path.basename(pth)); e = int(m.group(1)) if m else len(epochs)+1
        sd = torch.load(pth, map_location="cpu")
        A = extract_adaptive_A(sd)
        if A is None: continue
        # 快照图
        if e in [1,2,3] or (len(epoch_ckpts)>=5 and e % max(1, len(epoch_ckpts)//5)==0) or (pth==epoch_ckpts[-1]):
            plot_heatmap(A, f"A_adp (epoch {e})", os.path.join(figs_dir, f"A_adp_epoch{e}.png"))
            plot_heatmap(topk_mask(A, k=args.k_top), f"A_adp top-{args.k_top} (epoch {e})", os.path.join(figs_dir, f"A_adp_top{args.k_top}_epoch{e}.png"))
        # 统计
        diag_m.append(float(np.diag(A).mean()))
        offdiag_m.append(float((A.sum()-np.trace(A))/(A.size - A.shape[0])))
        asym_m.append(float(np.linalg.norm(A-A.T,'fro')/(np.linalg.norm(A,'fro')+1e-8)))
        specr_m.append(float(np.max(np.abs(np.linalg.eigvals(A)))))
        iu = np.triu_indices_from(A, k=1)
        from scipy.stats import spearmanr
        rho, _ = spearmanr(A[iu].ravel(), C[iu].ravel())
        simC_m.append(float(rho))
        epochs.append(e)

    if epochs:
        plot_series(epochs, {"diag_mean":diag_m, "offdiag_mean":offdiag_m, "asym_fro":asym_m}, os.path.join(figs_dir,"evolution_stat1.png"))
        plot_series(epochs, {"spectral_radius":specr_m, "spearman(A,C)":simC_m}, os.path.join(figs_dir,"evolution_stat2.png"))

    if best_ckpt:
        sd = torch.load(best_ckpt, map_location="cpu")
        A_best = extract_adaptive_A(sd)
        if A_best is not None:
            plot_heatmap(A_best, "A_adp (best)", os.path.join(figs_dir, "A_adp_best.png"))
            Cn = (C - C.min())/(C.max()-C.min()+1e-12)
            plot_heatmap(A_best - Cn, "A_best - Corr(norm)", os.path.join(figs_dir, "A_best_minus_corr.png"))

    # wandb 上传（*）
    if args.log_to_wandb and args.project and args.run_name:
        import wandb
        run = wandb.init(project=args.project, name=args.run_name, resume="allow")
        for fn in sorted(glob.glob(os.path.join(figs_dir, "*.png"))):
            run.log({"viz/"+os.path.basename(fn): wandb.Image(fn)})
        run.finish()

if __name__ == "__main__":
    main()
PY

# ========= 根据 DATA 自动推断保存目录与邻接路径（no JSON）=========
derive_paths () {
  local dat="${DATA^^}"    # upper
  if [[ "$dat" == *"FRANCE"* ]]; then
    SAVE_DIR="./garage/france/"
    ADJDATA="data/sensor_graph/adj_mx_france.pkl"
  elif [[ "$dat" == *"GERMANY"* ]]; then
    SAVE_DIR="./garage/germany/"
    ADJDATA="data/sensor_graph/adj_mx_germany.pkl"
  elif [[ "$dat" == *"METR"* ]]; then
    SAVE_DIR="./garage/metr/"
    ADJDATA="data/sensor_graph/adj_mx.pkl"
  elif [[ "$dat" == *"BAY"* ]]; then
    SAVE_DIR="./garage/bay/"
    ADJDATA="data/sensor_graph/adj_mx_bay.pkl"
  elif [[ "$dat" == *"SYNTHETIC_EASY"* ]]; then
    SAVE_DIR="./garage/synth_easy/"
    ADJDATA="data/sensor_graph/adj_mx_synthetic_easy.pkl"
  elif [[ "$dat" == *"SYNTHETIC_MEDIUM"* ]]; then
    SAVE_DIR="./garage/synth_medium/"
    ADJDATA="data/sensor_graph/adj_mx_synthetic_medium.pkl"
  elif [[ "$dat" == *"SYNTHETIC_HARD"* ]]; then
    SAVE_DIR="./garage/synth_hard/"
    ADJDATA="data/sensor_graph/adj_mx_synthetic_hard.pkl"
  elif [[ "$dat" == *"SYNTHETIC_VERY_HARD"* ]]; then
    SAVE_DIR="./garage/synth_very_hard/"
    ADJDATA="data/sensor_graph/adj_mx_synthetic_very_hard.pkl"
  else
    SAVE_DIR="./garage/"
    ADJDATA="data/sensor_graph/adj_mx.pkl"
  fi
  mkdir -p "$SAVE_DIR"
}

# ========= 跑一个实验的封装 =========
run_one () {
  local EXP_GROUP="$1"   # Baseline / PowerLaw / MixPropDual / Chebyshev / NoDiagonal
  local SEQ="$2"; local PRED="$3"; local LR="$4"; local DROPOUT="$5"; local NHID="$6"; local WD="$7"
  local EXP_NAME="${EXP_GROUP}_seq${SEQ}_pred${PRED}_lr${LR}_do${DROPOUT}_hid${NHID}_wd${WD}"

  derive_paths   # 得到 SAVE_DIR 和 ADJDATA

  local CFG_JSON
  CFG_JSON=$(cat <<JSON
{"data":"$DATA","device":"$DEVICE","epochs":$EPOCHS,"batch_size":$BATCH,
 "seq_length":$SEQ,"pred_length":$PRED,"learning_rate":$LR,"dropout":$DROPOUT,
 "nhid":$NHID,"weight_decay":$WD,"adjtype":"$ADJTYPE",
 "gcn_bool":true,"addaptadj":true,"randomadj":true,"print_every":$PRINT_EVERY}
JSON
)

  local CMD="python train.py \
    --data $DATA --device $DEVICE --batch_size $BATCH --epochs $EPOCHS \
    --seq_length $SEQ --pred_length $PRED \
    --learning_rate $LR --dropout $DROPOUT --nhid $NHID \
    --weight_decay $WD --print_every $PRINT_EVERY \
    --gcn_bool --addaptadj --randomadj --adjtype $ADJTYPE"

  echo ">>> [$EXP_NAME]"
  python _wandb_proxy.py --project "$WANDB_PROJECT" --name "$EXP_NAME" --config "$CFG_JSON" --cmd "$CMD"

  # === 训练后：可视化探针（不改 train.py）===
  if [[ "$LOG_TO_WANDB" == "1" ]]; then
    python _viz_probe.py \
      --save_dir "$SAVE_DIR" \
      --data "$DATA" \
      --adjdata "$ADJDATA" \
      --project "$WANDB_PROJECT" \
      --run_name "$EXP_NAME" \
      --log_to_wandb \
      --k_top "$K_TOP" || echo "[WARN] _viz_probe.py failed but training finished."
  else
    python _viz_probe.py \
      --save_dir "$SAVE_DIR" \
      --data "$DATA" \
      --adjdata "$ADJDATA" \
      --k_top "$K_TOP" || echo "[WARN] _viz_probe.py failed but training finished."
  fi
}

# ======================== 实验开关（按 EXP_ID 选择） ========================

# ---- 实验 1：Baseline ----
if [[ $EXP_ID -eq 0 || $EXP_ID -eq 1 ]]; then
  echo "==> EXP 1: Baseline"
  export GWN_USE_POWER=0; export GWN_USE_CHEBY=0; export GWN_DIAG_MODE=self_and_neighbor
  for SEQ in "${SEQ_LIST[@]}"; do for PRED in "${PRED_LIST[@]}"; do
    for LR in "${LR_LIST[@]}"; do for DROPOUT in "${DROPOUT_LIST[@]}"; do
      for NHID in "${NHID_LIST[@]}"; do for WD in "${WD_LIST[@]}"; do
        run_one "Baseline" "$SEQ" "$PRED" "$LR" "$DROPOUT" "$NHID" "$WD"
      done; done
    done; done
  done; done
fi

# ---- 实验 2：幂律传播 ----
if [[ $EXP_ID -eq 0 || $EXP_ID -eq 2 ]]; then
  echo "==> EXP 2: PowerLaw"
  export GWN_USE_POWER=1; export GWN_USE_CHEBY=0; export GWN_DIAG_MODE=self_and_neighbor
  for SEQ in "${SEQ_LIST[@]}"; do for PRED in "${PRED_LIST[@]}"; do
    for LR in "${LR_LIST[@]}"; do for DROPOUT in "${DROPOUT_LIST[@]}"; do
      for NHID in "${NHID_LIST[@]}"; do for WD in "${WD_LIST[@]}"; do
        run_one "PowerLaw" "$SEQ" "$PRED" "$LR" "$DROPOUT" "$NHID" "$WD"
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
  for SEQ in "${SEQ_LIST[@]}"; do for PRED in "${PRED_LIST[@]}"; do
    for LR in "${LR_LIST[@]}"; do for DROPOUT in "${DROPOUT_LIST[@]}"; do
      for NHID in "${NHID_LIST[@]}"; do for WD in "${WD_LIST[@]}"; do
        run_one "MixPropDual" "$SEQ" "$PRED" "$LR" "$DROPOUT" "$NHID" "$WD"
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
  for SEQ in "${SEQ_LIST[@]}"; do for PRED in "${PRED_LIST[@]}"; do
    for LR in "${LR_LIST[@]}"; do for DROPOUT in "${DROPOUT_LIST[@]}"; do
      for NHID in "${NHID_LIST[@]}"; do for WD in "${WD_LIST[@]}"; do
        run_one "Chebyshev" "$SEQ" "$PRED" "$LR" "$DROPOUT" "$NHID" "$WD"
      done; done
    done; done
  done; done
fi

# ---- 实验 5：无对角邻接 ----
if [[ $EXP_ID -eq 0 || $EXP_ID -eq 5 ]]; then
  echo "==> EXP 5: NoDiagonal"
  export GWN_USE_POWER=0; export GWN_USE_CHEBY=0; export GWN_DIAG_MODE=neighbor
  for SEQ in "${SEQ_LIST[@]}"; do for PRED in "${PRED_LIST[@]}"; do
    for LR in "${LR_LIST[@]}"; do for DROPOUT in "${DROPOUT_LIST[@]}"; do
      for NHID in "${NHID_LIST[@]}"; do for WD in "${WD_LIST[@]}"; do
        run_one "NoDiagonal" "$SEQ" "$PRED" "$LR" "$DROPOUT" "$NHID" "$WD"
      done; done
    done; done
  done; done
fi

echo "✅ 实验完成（EXP_ID=$EXP_ID）。wandb 项目：$WANDB_PROJECT"



# SYNTHETIC_EASY / SYNTHETIC_MEDIUM / SYNTHETIC_HARD / SYNTHETIC_VERY_HARD

# WANDB_PROJECT=GWN-Grid-SYNTHETIC_EASY_4 DATA=data/SYNTHETIC_EASY DEVICE=cpu EPOCHS=1 EXP_ID=4 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-SYNTHETIC_EASY_1 DATA=data/SYNTHETIC_EASY DEVICE=cuda:0 EPOCHS=50 EXP_ID=1 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-SYNTHETIC_EASY_2 DATA=data/SYNTHETIC_EASY DEVICE=cuda:0 EPOCHS=50 EXP_ID=2 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-SYNTHETIC_EASY_3 DATA=data/SYNTHETIC_EASY DEVICE=cuda:0 EPOCHS=50 EXP_ID=3 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-SYNTHETIC_EASY_4 DATA=data/SYNTHETIC_EASY DEVICE=cuda:0 EPOCHS=50 EXP_ID=4 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-SYNTHETIC_EASY_5 DATA=data/SYNTHETIC_EASY DEVICE=cuda:0 EPOCHS=50 EXP_ID=5 bash run_experiments.sh



# WANDB_PROJECT=GWN-Grid-SYNTHETIC_HARD_1 DATA=data/SYNTHETIC_HARD DEVICE=cuda:0 EPOCHS=50 EXP_ID=1 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-SYNTHETIC_HARD_2 DATA=data/SYNTHETIC_HARD DEVICE=cuda:0 EPOCHS=50 EXP_ID=2 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-SYNTHETIC_HARD_3 DATA=data/SYNTHETIC_HARD DEVICE=cuda:0 EPOCHS=50 EXP_ID=3 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-SYNTHETIC_HARD_4 DATA=data/SYNTHETIC_HARD DEVICE=cuda:0 EPOCHS=50 EXP_ID=4 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-SYNTHETIC_HARD_5 DATA=data/SYNTHETIC_HARD DEVICE=cuda:0 EPOCHS=50 EXP_ID=5 bash run_experiments.sh



# FRANCE / GERMANY

# WANDB_PROJECT=GWN-Grid-FRANCE_1 DATA=data/FRANCE DEVICE=cuda:0 EPOCHS=5 EXP_ID=1 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-FRANCE_2 DATA=data/FRANCE DEVICE=cuda:0 EPOCHS=5 EXP_ID=2 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-FRANCE_3 DATA=data/FRANCE DEVICE=cuda:0 EPOCHS=5 EXP_ID=3 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-FRANCE_4 DATA=data/FRANCE DEVICE=cuda:0 EPOCHS=5 EXP_ID=4 bash run_experiments.sh

# WANDB_PROJECT=GWN-Grid-FRANCE_5 DATA=data/FRANCE DEVICE=cuda:0 EPOCHS=5 EXP_ID=5 bash run_experiments.sh