
"""
输出目录: outputs_dynamic_corr/compare/
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------ 基础 I/O 与计算 ------------
def load_timeseries_csv(path, drop_first_col=False):
    df = pd.read_csv(path)

    # 1) 优先删除名为 Time_Step 的列
    if "Time_Step" in df.columns:
        df = df.drop(columns=["Time_Step"])

    # 2) option：强制丢弃首列（若首列为时间戳但未命名为 Time_Step）
    if drop_first_col and df.shape[1] >= 2:
        df = df.iloc[:, 1:]

    # 3) 尝试将剩余列转换为数值；无法转换的整列丢弃
    num_df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')

    if num_df.shape[1] == 0:
        raise ValueError(f"{path}: 删除 Time_Step/首列后没有可用的数值列。")

    X = num_df.values  # (T, N)
    return X

def corr_matrix(X_window, method="pearson"):
    if method == "pearson":
        C = np.corrcoef(X_window, rowvar=False)
        return np.clip(C, -1.0, 1.0)
    elif method == "cosine":
        X = X_window.astype(float)
        X = X - X.mean(axis=0, keepdims=True)
        denom = np.linalg.norm(X, axis=0, keepdims=True) + 1e-12
        Xn = X / denom
        C = Xn.T @ Xn
        return np.clip(C, -1.0, 1.0)
    else:
        raise ValueError(f"Unsupported method: {method}")

def rolling_corr(X, window=24, step=1, method="pearson"):
    T, _ = X.shape
    mats, idx = [], []
    for s in range(0, T - window + 1, step):
        seg = X[s:s+window]
        mats.append(corr_matrix(seg, method=method))
        idx.append(s + window - 1)
    return np.stack(mats, axis=0), np.array(idx)

def topk_adjacency(C, k=3, symmetrize=True):
    N = C.shape[0]
    A = np.zeros_like(C, dtype=float)
    M = C.copy()
    np.fill_diagonal(M, -np.inf)
    idx = np.argpartition(-np.abs(M), kth=k, axis=1)[:, :k]
    rows = np.repeat(np.arange(N), k)
    cols = idx.reshape(-1)
    A[rows, cols] = 1.0
    if symmetrize:
        A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return A

def threshold_adjacency(C, thr=0.7, symmetrize=True):
    M = C.copy()
    np.fill_diagonal(M, 0.0)
    A = (np.abs(M) >= thr).astype(float)
    if symmetrize:
        A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return A

# ------------ 可视化辅助 ------------
def ensure_outdirs(*ds):
    for d in ds:
        os.makedirs(d, exist_ok=True)

def dataset_label_from_path(path):
    base = os.path.basename(path)
    parent = os.path.basename(os.path.dirname(path))
    name = os.path.splitext(base)[0]
    label = parent if parent else name
    if "SYNTHETIC" in label.upper():
        return label.replace("_", " ").title()
    return name

import matplotlib.pyplot as plt

def plot_2x2_heatmaps(mats, titles, suptitle, outpath, vmin=-1, vmax=1):
    # Let Matplotlib manage spacing intelligently
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes = axes.ravel()

    im = None
    for M, ax, title in zip(mats, axes, titles):
        im = ax.imshow(M, aspect='equal', interpolation='none', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Node")
        ax.set_ylabel("Node")

        # Keep each image centered and square
        ax.set_anchor('C')          # center the artist within its axes
        try:
            ax.set_box_aspect(1)    # requires Matplotlib >= 3.3
        except Exception:
            pass

    # One shared colorbar on the LEFT. Constrained layout keeps it from overlapping.
    cbar = fig.colorbar(
        im,
        ax=axes.tolist(),
        location="left",
        pad=0.02,        # distance between cbar and subplots
        fraction=0.05,   # thickness of the colorbar relative to the subplot group
        shrink=0.95      # slightly shorter so it looks balanced vertically
    )
    cbar.ax.set_ylabel("Correlation", rotation=90, labelpad=12)

    fig.suptitle(suptitle, fontsize=13)

    # No need for tight_layout when using constrained_layout
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_variance_2x2(mats, titles, suptitle, outpath):
    vmaxs = [np.percentile(m, 99) if m.max() > 0 else 1.0 for m in mats]
    vmax = max(vmaxs) if np.isfinite(max(vmaxs)) else 1.0
    plot_2x2_heatmaps(mats, titles, suptitle, outpath, vmin=0, vmax=vmax)

import numpy as np
import matplotlib.pyplot as plt

def plot_montage_compare(
    stacks, labels, slices=4, outpath="rolling_corr_montage_compare.png",
    vmin=-1, vmax=1, suptitle="Rolling Correlation Heatmaps — Comparative Montage"
):
    """
    stacks: list of 3D arrays, each with shape (K, N, N)
    labels: list of strings, one per stack (row label)
    slices: number of slices (columns) to show per stack
    """

    rows = len(stacks)
    cols = max(1, int(slices))
    assert len(labels) == rows, "labels must match number of stacks"

    # Let Matplotlib manage spacing (prevents cbar/subplot overlap).
    fig, axes = plt.subplots(
        rows, cols, figsize=(4*cols, 3.6*rows), squeeze=False, constrained_layout=True
    )

    im = None
    for r in range(rows):
        stack = stacks[r]
        K = stack.shape[0] if stack is not None and hasattr(stack, "shape") and len(stack.shape) == 3 else 0

        # Choose slice indices evenly across [0, K-1].
        if K > 0:
            picks = np.linspace(0, K-1, cols, dtype=int)
        else:
            picks = []

        for c in range(cols):
            ax = axes[r, c]
            if K == 0:
                ax.axis('off')
                if c == 0:
                    ax.set_title("No data")
                    ax.set_ylabel(labels[r])
                continue

            im = ax.imshow(
                stack[picks[c]], aspect='equal', interpolation='none',
                vmin=vmin, vmax=vmax
            )

            # Keep each heatmap centered and square even if figure resizes.
            ax.set_anchor('C')
            try:
                ax.set_box_aspect(1)   # mpl >= 3.3
            except Exception:
                pass

            # Row label on the first column only.
            if c == 0:
                ax.set_ylabel(labels[r])

            ax.set_title(f"slice {c+1}/{cols} (k={picks[c]})", fontsize=10)
            ax.set_xlabel("Node")
            ax.set_ylabel("Node")

    # Only add a colorbar if we plotted at least one image.
    if im is not None:
        cbar = fig.colorbar(
            im, ax=axes.ravel().tolist(),
            location="left", pad=0.02, fraction=0.05, shrink=0.95
        )
        cbar.ax.set_ylabel("Correlation", rotation=90, labelpad=12)

    fig.suptitle(suptitle, fontsize=13)

    # With constrained_layout we shouldn't also call tight_layout; saving with bbox_inches tight is fine.
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
def plot_multi_lines(series_list, labels, title, ylabel, outpath):
    plt.figure(figsize=(9,4))
    for y, lab in zip(series_list, labels):
        if len(y) > 0:
            plt.plot(y, label=lab)
    plt.xlabel("Window Index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


import numpy as np
import matplotlib.pyplot as plt

def plot_time_series_windows(data, window=24, outpath="time_series_windows.png"):
    """
    Visualize multivariate time series in fixed windows.
    
    Args:
        data: numpy array of shape [T, N], where T=temporal length, N=number of variables
        window: int, length of each time window
        outpath: str, file path to save the figure
    """
    T, N = data.shape
    n_windows = int(np.ceil(T / window))

    fig, axes = plt.subplots(N, 1, figsize=(12, 2.5 * N), sharex=True)
    if N == 1:
        axes = [axes]

    for j in range(N):
        ax = axes[j]
        for w in range(n_windows):
            start, end = w * window, min((w + 1) * window, T)
            ax.plot(
                np.arange(start, end), data[start:end, j],
                label=f"Window {w+1}" if j == 0 else None
            )
        ax.set_ylabel(f"Var {j+1}")
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Time")
    if N > 1:
        axes[0].legend(loc="upper right", ncol=4, fontsize=8)

    fig.suptitle(f"Multivariate Time Series (window={window})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

# ------------ 主流程 ------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs=4, help="四个CSV文件路径(多变量时间序列)")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--method", type=str, default="pearson", choices=["pearson", "cosine"])
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--montage_slices", type=int, default=4, help="每个数据集展示的时间片数")
    parser.add_argument("--drop_first_col", action="store_true",
                        help="除 Time_Step 外，额外强制丢弃首列（若首列为时间戳但未命名为 Time_Step）")
    args = parser.parse_args()

    outdir = os.path.join("outputs_dynamic_corr", "compare")
    ensure_outdirs(outdir)

    # 读取与预计算
    labels = [dataset_label_from_path(p) for p in args.paths]
    Xs = [load_timeseries_csv(p, drop_first_col=args.drop_first_col) for p in args.paths]
    plot_time_series_windows(Xs[0], window=24, outpath="mts_windows.png")

    # 1) 全时段相关矩阵
    C_full = [corr_matrix(X, method=args.method) for X in Xs]

    # 2) 滚动相关
    stacks, time_idxs = [], []
    for X in Xs:
        if X.shape[0] < args.window:
            stacks.append(np.zeros((0, 1, 1)))
            time_idxs.append(np.array([]))
        else:
            S, idx = rolling_corr(X, window=args.window, step=args.step, method=args.method)
            stacks.append(S)
            time_idxs.append(idx)

    # 3) 方差矩阵（跨时间）
    var_mats = []
    for S in stacks:
        if S.shape[0] == 0:
            var_mats.append(np.zeros((1,1)))
        else:
            V = S.var(axis=0)
            np.fill_diagonal(V, 0.0)
            var_mats.append(V)

    # 4) 相邻窗口差的弗罗贝尼乌斯范数曲线
    fro_deltas = []
    for S in stacks:
        if S.shape[0] <= 1:
            fro_deltas.append(np.array([]))
        else:
            diffs = []
            for k in range(1, S.shape[0]):
                D = S[k] - S[k-1]
                np.fill_diagonal(D, 0.0)
                diffs.append(np.linalg.norm(D, 'fro'))
            fro_deltas.append(np.array(diffs))

    # 5) 边密度曲线（Top-K / Thr）
    def edge_density_series(S, builder):
        if S.shape[0] == 0:
            return np.array([])
        N = S.shape[1]
        denom = N*(N-1)/2.0 + 1e-12
        dens = []
        for k in range(S.shape[0]):
            A = builder(S[k])
            m = np.triu(A, 1).sum()
            dens.append(float(m)/denom)
        return np.array(dens)

    dens_topk = [edge_density_series(S, lambda C: topk_adjacency(C, k=args.topk, symmetrize=True)) for S in stacks]
    dens_thr  = [edge_density_series(S, lambda C: threshold_adjacency(C, thr=args.threshold, symmetrize=True)) for S in stacks]

    # ------------ 生成对比图 ------------
    plot_2x2_heatmaps(
        C_full, labels,
        suptitle=f"Full-period Correlation ({args.method}) — Comparative",
        outpath=os.path.join(outdir, "static_corr_compare.png"),
        vmin=-1, vmax=1
    )

    plot_montage_compare(
        stacks, labels, slices=args.montage_slices,
        outpath=os.path.join(outdir, "rolling_corr_montage_compare.png")
    )

    plot_variance_2x2(
        var_mats, labels,
        suptitle="Variance of Rolling Correlation (edge-wise) — Comparative",
        outpath=os.path.join(outdir, "corr_variance_compare.png")
    )

    plot_multi_lines(
        fro_deltas, labels,
        title="Change of Correlation Over Time (Frobenius Δ, adjacent windows)",
        ylabel="Frobenius Δ",
        outpath=os.path.join(outdir, "corr_change_over_time_compare.png")
    )

    plot_multi_lines(
        dens_topk, labels,
        title=f"Edge Density Over Time (Top-{args.topk}) — Comparative",
        ylabel="Edge Density",
        outpath=os.path.join(outdir, "edge_density_topk_compare.png")
    )

    plot_multi_lines(
        dens_thr, labels,
        title=f"Edge Density Over Time (|corr| ≥ {args.threshold:g}) — Comparative",
        ylabel="Edge Density",
        outpath=os.path.join(outdir, "edge_density_thr_compare.png")
    )

    print(f"[OK] All comparative figures saved to: {outdir}")

if __name__ == "__main__":
    main()






# python dynamic_corr_compare.py \
#   --window 24 --step 1 --method pearson \
#   --topk 3 --threshold 0.7 --montage_slices 4 \
#   "/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/GNN/Graph-WaveNet-master-origin/data/SYNTHETIC_EASY/synthetic_time_series.csv" \
#   "/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/GNN/Graph-WaveNet-master-origin/data/SYNTHETIC_MEDIUM/synthetic_time_series.csv" \
#   "/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/GNN/Graph-WaveNet-master-origin/data/SYNTHETIC_HARD/synthetic_time_series.csv" \
#   "/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/GNN/Graph-WaveNet-master-origin/data/SYNTHETIC_VERY_HARD/synthetic_time_series.csv"
