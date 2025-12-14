# cat > _wandb_proxy.py <<'PY'
import os, re, sys, json, shlex, subprocess, csv, time
import wandb

def ensure_csv_header(csv_path, fieldnames):
    exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None
    if not exists:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()

def append_csv(csv_path, row, fieldnames):
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writerow(row)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", required=True, help="json string")
    ap.add_argument("--cmd", required=True, help="command to run")
    args = ap.parse_args()

    cfg = json.loads(args.config)
    run = wandb.init(project=args.project, name=args.name, config=cfg, dir=os.environ.get("WANDB_DIR","./wandb_runs"))
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("valid/*", step_metric="epoch")
    wandb.define_metric("test/*",  step_metric="epoch")

    gwn_flags = {k:v for k,v in os.environ.items() if k.startswith("GWN_")}
    wandb.config.update({"env_flags": gwn_flags}, allow_val_change=True)

    proc = subprocess.Popen(
        shlex.split(args.cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, universal_newlines=True
    )

    re_iter   = re.compile(r"Iter:\s*(\d+),\s*Train Loss:\s*([\d\.eE+-]+),\s*Train MAPE:\s*([\d\.eE+-]+),\s*Train RMSE:\s*([\d\.eE+-]+)")
    re_epoch  = re.compile(r"Epoch:\s*(\d+),.*Valid Loss:\s*([\d\.eE+-]+),\s*Valid MAPE:\s*([\d\.eE+-]+),\s*Valid RMSE:\s*([\d\.eE+-]+)")
    re_epoch_time = re.compile(r"Epoch:\s*(\d+),\s*Inference Time:")
    re_best   = re.compile(r"The valid loss on best model is\s*([\d\.eE+-]+)")
    re_hz     = re.compile(r"Evaluate best model on test data for horizon\s*(\d+),\s*Test MAE:\s*([\d\.eE+-]+),\s*Test MAPE:\s*([\d\.eE+-]+),\s*Test RMSE:\s*([\d\.eE+-]+)")
    re_avg    = re.compile(r"On average over\s*(\d+)\s*horizons,\s*Test MAE:\s*([\d\.eE+-]+),\s*Test MAPE:\s*([\d\.eE+-]+),\s*Test RMSE:\s*([\d\.eE+-]+)")

    csv_fields = [
        "exp_group","dataset","batch_size","seq","pred","lr","dropout","nhid","wd",
        "blocks","layers",
        "avg_test_mae","avg_test_rmse","avg_test_mape","best_valid_loss","run_name","timestamp"
    ]
    results_csv = os.environ.get("RESULTS_CSV","./results.csv")
    ensure_csv_header(results_csv, csv_fields)

    current_epoch = 0
    best_valid = None
    avg_mae = avg_rmse = avg_mape = None

    for line in proc.stdout:
        sys.stdout.write(line); sys.stdout.flush()

        m = re_epoch_time.search(line)
        if m:
            current_epoch = int(m.group(1)); wandb.log({"epoch": current_epoch}, step=current_epoch); continue

        m = re_iter.search(line)
        if m:
            wandb.log({"epoch": current_epoch,
                       "train/loss": float(m.group(2)),
                       "train/mape": float(m.group(3)),
                       "train/rmse": float(m.group(4))}, step=current_epoch); continue

        m = re_epoch.search(line)
        if m:
            current_epoch = int(m.group(1))
            wandb.log({"epoch": current_epoch,
                       "valid/loss": float(m.group(2)),
                       "valid/mape": float(m.group(3)),
                       "valid/rmse": float(m.group(4))}, step=current_epoch); continue

        m = re_best.search(line)
        if m:
            best_valid = float(m.group(1)); wandb.summary["best_valid/loss"] = best_valid; continue

        m = re_hz.search(line)
        if m:
            hz = int(m.group(1))
            wandb.log({"epoch": current_epoch,
                       "test/horizon": hz,
                       "test/horizon_mae":  float(m.group(2)),
                       "test/horizon_mape": float(m.group(3)),
                       "test/horizon_rmse": float(m.group(4))}, step=current_epoch); continue

        m = re_avg.search(line)
        if m:
            wandb.summary["test/avg_horizons"] = int(m.group(1))
            avg_mae  = float(m.group(2)); avg_mape = float(m.group(3)); avg_rmse = float(m.group(4))
            wandb.summary["test/avg_mae"]  = avg_mae
            wandb.summary["test/avg_mape"] = avg_mape
            wandb.summary["test/avg_rmse"] = avg_rmse
            continue

    proc.wait()

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
        "blocks":    cfg.get("blocks",""),
        "layers":    cfg.get("layers",""),
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
# PY




'''
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
        "exp_group","model_name","dataset","batch_size","seq","pred","lr","dropout","nhid","wd",
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

    # ---- 训练完成：结果 CSV ----
    # 从 config 读取参数（带安全默认）
    cfg = dict(wandb.config)
    row = {
        "exp_group": cfg.get("exp_group",""),
        "model_name": os.environ.get("MODEL_NAME",""),   # 👈 新增
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
'''