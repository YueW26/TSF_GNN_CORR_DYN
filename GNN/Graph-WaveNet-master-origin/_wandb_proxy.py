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
