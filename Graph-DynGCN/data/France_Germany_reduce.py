# France_Germany_reduce.py
import os
import time
import math
import numpy as np
import pandas as pd

# ============== GPU (PyTorch) ==============
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ----------------- 进度打印 -----------------
def _progress_bar(cur, total, prefix=""):
    pct = 100.0 * cur / max(1, total)
    bar_len = 30
    filled = int(bar_len * pct / 100.0)
    bar = "█" * filled + "-" * (bar_len - filled)
    print(f"\r{prefix}[{bar}] {pct:6.2f}% ({cur}/{total})", end="", flush=True)

def _stage(msg):
    print(msg, flush=True)

# ----------------- GPU: 批量窗口 Pearson 相关 -----------------
def _corr_from_windows_gpu(Wb):  # Wb: (B, window, N) float32 on CUDA/CPU
    mu  = Wb.mean(dim=1, keepdim=True)                     # (B,1,N)
    Z   = Wb - mu
    std = Z.std(dim=1, unbiased=True, keepdim=True)        # (B,1,N)
    std = torch.clamp(std, min=1e-6)
    Z   = Z / std                                          # z-score
    A   = torch.bmm(Z.transpose(1,2), Z) / max(1, Wb.shape[1]-1)  # (B,N,N)
    A   = torch.clamp(A, -1.0, 1.0)
    I   = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype).unsqueeze(0)
    A   = A * (1.0 - I)                                    # 去自环
    return A

def tgv_series_streaming_gpu(X_np, window=24, step=10, batch_size=512, device="cuda",
                             verbose_prefix="TGV(full)  "):
    T, N = X_np.shape
    if T - window + 1 < 2:
        return np.zeros((0,), dtype=np.float32)

    X = torch.from_numpy(X_np.astype(np.float32)).to(device, non_blocking=True)
    W = X.unfold(dimension=0, size=window, step=step)  # (num_win, window, N)
    num_win = W.shape[0]
    if num_win < 2:
        return np.zeros((0,), dtype=np.float32)

    tgvs = []
    prevA = None
    last_print = -1

    for start in range(0, num_win, batch_size):
        end = min(start + batch_size, num_win)
        Ab = _corr_from_windows_gpu(W[start:end])           # (B,N,N)

        if prevA is None:
            if Ab.shape[0] >= 2:
                d = Ab[1:] - Ab[:-1]
                tgvs.append(torch.linalg.norm(d.reshape(d.shape[0], -1), dim=1))
        else:
            d0 = Ab[0] - prevA
            tgvs.append(torch.linalg.norm(d0.reshape(1, -1), dim=1))
            if Ab.shape[0] >= 2:
                d = Ab[1:] - Ab[:-1]
                tgvs.append(torch.linalg.norm(d.reshape(d.shape[0], -1), dim=1))

        prevA = Ab[-1].detach()

        # 进度条
        processed = min(end, num_win)
        done = min(num_win - 1, processed - 1)
        if done != last_print and (done == num_win - 1 or done % max(1, (num_win - 1)//20) == 0):
            _progress_bar(done, num_win - 1, prefix=verbose_prefix)
            last_print = done

        del Ab
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)

    out = torch.cat(tgvs, dim=0) if len(tgvs) else torch.empty(0, dtype=torch.float32)
    _progress_bar(num_win - 1, num_win - 1, prefix=verbose_prefix)
    print("", flush=True)
    return out.detach().cpu().numpy()

# ============== 采样（时间公平 + TGV 等质量） ==============
def _round_with_budget(arr, K):
    arr  = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    base = np.floor(arr).astype(int)
    rem  = int(K - base.sum())
    if rem > 0:
        frac = arr - base
        idx  = np.argsort(-frac)[:rem]
        base[idx] += 1
    elif rem < 0:
        frac = arr - base
        idx  = np.argsort(frac)[:(-rem)]
        base[idx] -= 1
    return base

def fair_tgv_sampling_indices(TGV_t, K, M=12, alpha=0.5, timeout_sec=5.0):
    """
    双约束采样；若超过 timeout_sec 未完成，回退到均匀采样。
    """
    t_start = time.time()
    Tm1 = len(TGV_t)
    if Tm1 == 0:
        return []
    if K >= Tm1:
        return list(range(Tm1))

    # 分箱
    edges = np.linspace(0, Tm1, M + 1, dtype=int)
    L = np.diff(edges)
    S = np.array([TGV_t[edges[b]:edges[b+1]].sum() for b in range(M)], dtype=float)

    # 权重
    Ssum = S.sum()
    if Ssum <= 1e-12:
        weights = (L / max(L.sum(), 1))
    else:
        weights = (1 - alpha) * (L / max(L.sum(), 1)) + alpha * (S / Ssum)

    k = _round_with_budget(weights * K, K)

    # 调和为 K
    if k.sum() < K:
        # 按 S/(k+eps) 最大的补
        while k.sum() < K:
            if time.time() - t_start > timeout_sec:
                # 回退：均匀采样
                step = max(1, Tm1 // K)
                return list(range(0, Tm1, step))[:K]
            b = int(np.argmax(S / (k + 1e-12)))
            k[b] += 1
    elif k.sum() > K:
        while k.sum() > K:
            if time.time() - t_start > timeout_sec:
                step = max(1, Tm1 // K)
                return list(range(0, Tm1, step))[:K]
            b = int(np.argmax(k))
            k[b] -= 1

    # 箱内等质量
    selected = []
    for b in range(M):
        if time.time() - t_start > timeout_sec:
            step = max(1, Tm1 // K)
            return list(range(0, Tm1, step))[:K]

        if k[b] <= 0 or L[b] <= 0:
            continue
        start, end = edges[b], edges[b + 1]
        seg = TGV_t[start:end]
        S_b = seg.sum()
        if S_b <= 1e-12:
            # 该箱 TGV 很小：均匀取
            idxs = np.linspace(start, end - 1, num=k[b], dtype=int)
            selected.extend(idxs.tolist())
        else:
            q = S_b / k[b]
            acc, cnt = 0.0, 0
            # 为了避免极端长循环，加入最大迭代保护
            for it, t in enumerate(range(start, end)):
                acc += TGV_t[t]
                if acc >= q - 1e-12:
                    selected.append(t)
                    acc, cnt = 0.0, cnt + 1
                    if cnt == k[b]:
                        break
                if it % 10000 == 0 and it > 0:
                    if time.time() - t_start > timeout_sec:
                        break
            # 若不足，均匀补齐
            if cnt < k[b]:
                extra = np.linspace(start, end - 1, num=(k[b] - cnt), dtype=int)
                selected.extend(extra.tolist())

    selected = sorted(set(int(x) for x in selected))
    if len(selected) > K:
        selected = selected[:K]
    # 万一还不够，均匀补齐
    while len(selected) < K:
        if time.time() - t_start > timeout_sec:
            break
        missing = K - len(selected)
        add = np.linspace(0, Tm1 - 1, num=missing, dtype=int).tolist()
        selected = sorted(set(selected + add))
        if len(selected) > K:
            selected = selected[:K]
            break

    return selected[:K]

# ============== 主流程：读取 -> TGV -> 采样 -> 写文件 -> 验证 ==============
def compress_dataset(path,
                     window=24,
                     ratio=0.1,
                     M=12,
                     alpha=0.5,
                     out_suffix="_reduced.csv",
                     use_gpu=True,
                     step=10,
                     batch_size=512,
                     pre_downsample=1,
                     sampling_timeout=5.0,
                     debug=True):
    print(f"\n=== Processing {path} ===", flush=True)
    t0 = time.time()
    df = pd.read_csv(path)

    # 数值列
    df_num = df.select_dtypes(include=[np.number]).interpolate(limit_direction="both").dropna()
    if pre_downsample > 1:
        df_num = df_num.iloc[::pre_downsample, :].reset_index(drop=True)

    X = df_num.values.astype(np.float32)
    T, N = X.shape
    print(f"  Loaded numeric matrix shape: (T={T}, N={N}), pre_downsample={pre_downsample}", flush=True)

    # 自动缩窗
    if T - window + 1 < 2:
        window = max(3, min(T - 2, max(3, T // 10)))
        print(f"  [auto] shrink window to {window} due to short length.", flush=True)

    # 设备
    device = "cuda" if (use_gpu and TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
    if device == "cuda":
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}  (step={step}, batch={batch_size})", flush=True)
    else:
        print(f"  Using CPU (PyTorch available={TORCH_AVAILABLE}) (step={step}, batch={batch_size})", flush=True)

    # ---- TGV(full) ----
    _stage("  [stage] computing TGV(full)...")
    t1 = time.time()
    tgv_full = tgv_series_streaming_gpu(
        X, window=window, step=step, batch_size=batch_size, device=device, verbose_prefix="TGV(full)  "
    )
    TGV_total_full = float(tgv_full.sum()) if tgv_full.size > 0 else 0.0
    print(f"  TGV(full) length: {len(tgv_full)}  total={TGV_total_full:.6f}  (cost {time.time()-t1:.2f}s)", flush=True)

    # ---- 采样 ----
    _stage("  [stage] sampling indices...")
    K_target = max(1, int(len(tgv_full) * ratio))
    try:
        idx = fair_tgv_sampling_indices(tgv_full, K_target, M=M, alpha=alpha, timeout_sec=sampling_timeout)
    except Exception as e:
        print(f"  [warn] fair sampling failed: {e}; fallback to uniform.", flush=True)
        step_u = max(1, len(tgv_full) // K_target)
        idx = list(range(0, len(tgv_full), step_u))[:K_target]

    print(f"  Sampling: target={K_target}  selected={len(idx)}  boxes(M)={M}  alpha={alpha}", flush=True)

    # ---- 映射到原始行 ----
    _stage("  [stage] mapping indices to rows...")
    if len(idx) == 0:
        rows = [min(window, len(df) - 1)]
    else:
        rows_dfnum = [min(int(t + window), len(df_num) - 1) for t in idx]
        rows = [min(int(r * pre_downsample), len(df) - 1) for r in rows_dfnum]
        rows = sorted(set(rows))
    print(f"  rows mapped: {len(rows)}", flush=True)

    # ---- 写文件：先落地到 /tmp 再 cp 回网络盘（避免网盘 I/O 卡住）----
    _stage("  [stage] writing csv...")
    df_reduced = df.iloc[rows].reset_index(drop=True)
    out_path = path.replace(".csv", out_suffix)
    tmp_out  = f"/tmp/{os.path.basename(out_path)}"
    try:
        df_reduced.to_csv(tmp_out, index=False)
        os.system(f"cp {tmp_out} {out_path}")
        os.remove(tmp_out) if os.path.exists(tmp_out) else None
        print(f"  [ok] written to {out_path}", flush=True)
    except Exception as e:
        print(f"  [warn] tmp write failed: {e}; writing directly to {out_path}", flush=True)
        df_reduced.to_csv(out_path, index=False)

    # ---- TGV(reduced) 校验 ----
    _stage("  [stage] validating TGV(reduced)...")
    df_red_num = df_reduced.select_dtypes(include=[np.number]).interpolate(limit_direction="both").dropna()
    Xr = df_red_num.values.astype(np.float32)
    Tr, Nr = Xr.shape
    window_r = window
    if Tr - window_r + 1 < 2:
        window_r = max(3, min(Tr - 2, max(3, Tr // 10)))
        print(f"  [auto] shrink reduced-window to {window_r}", flush=True)

    # reduced 步长---稍微小一点，避免空序列
    step_r = min(step, max(1, (Tr - window_r) // 500 + 1))
    t2 = time.time()
    tgv_red = tgv_series_streaming_gpu(
        Xr, window=window_r, step=step_r, batch_size=batch_size, device=device, verbose_prefix="TGV(reduced)"
    )
    TGV_total_red = float(tgv_red.sum()) if tgv_red.size > 0 else 0.0
    print(f"  TGV(reduced) length: {len(tgv_red)}  total={TGV_total_red:.6f}  (cost {time.time()-t2:.2f}s)", flush=True)

    # ---- 汇总 ----
    print(f"✅ Saved to {out_path}", flush=True)
    print(f"  Variables (columns, numeric): {Nr}", flush=True)
    print(f"  Timestamps (rows):            {df_reduced.shape[0]}", flush=True)
    print(f"  TGV total (before):           {TGV_total_full:.6f}", flush=True)
    print(f"  TGV total (after):            {TGV_total_red:.6f}", flush=True)
    print(f"⏱️  Total cost: {time.time()-t0:.2f}s", flush=True)

    return out_path

# ============== 执行 ==============
if __name__ == "__main__":
    paths = [
        "/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/GNN/Graph-WaveNet-master-origin/data/France_processed_0.csv",
        "/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/GNN/Graph-WaveNet-master-origin/data/Germany_processed_0.csv",
    ]
    for p in paths:
        compress_dataset(
            p,
            window=24,
            ratio=0.1,
            M=24,                 # 更细分箱，时间覆盖更均匀
            alpha=0.5,
            out_suffix="_reduced.csv",
            use_gpu=True,
            step=10,              # 50快，10慢
            batch_size=1024,      # 
            pre_downsample=1,     # 如 I/O 瓶颈 --- 设为 2 或 3
            sampling_timeout=5.0, # 采样超时 5s 回退
            debug=True
        )
        



