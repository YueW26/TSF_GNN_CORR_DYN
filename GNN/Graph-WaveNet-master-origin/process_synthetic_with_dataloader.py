import os
import shutil

# 目标根目录（按你的路径）
ROOT = "/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/GNN/Graph-WaveNet-master-origin"
TARGET_DIRS = [
    os.path.join(ROOT, "data", "synthetic"),
    os.path.join(ROOT, "data"),
]

SUBSTR = "tgv"          # 不区分大小写匹配
CASE_INSENSITIVE = True

def contains_tgv(name: str) -> bool:
    return (SUBSTR.lower() in name.lower()) if CASE_INSENSITIVE else (SUBSTR in name)

def delete_items_with_substring(root: str):
    if not os.path.isdir(root):
        print(f"[skip] 不存在目录：{root}")
        return 0, 0

    # 收集待删的文件与目录（先收集，再删除，避免遍历过程中修改目录树）
    to_delete_files = []
    to_delete_dirs  = []

    for dirpath, dirnames, filenames in os.walk(root):
        # 目录命中
        for d in dirnames:
            if contains_tgv(d):
                to_delete_dirs.append(os.path.join(dirpath, d))
        # 文件命中
        for f in filenames:
            if contains_tgv(f):
                to_delete_files.append(os.path.join(dirpath, f))

    # 为避免父目录先删导致路径失效，目录按深度从深到浅删除
    to_delete_dirs = sorted(set(to_delete_dirs), key=lambda p: p.count(os.sep), reverse=True)
    to_delete_files = sorted(set(to_delete_files))

    # 执行删除
    deleted_files = 0
    deleted_dirs  = 0

    for f in to_delete_files:
        try:
            os.remove(f)
            print(f"[file] 删除：{f}")
            deleted_files += 1
        except Exception as e:
            print(f"[file] 无法删除：{f}  -> {e}")

    for d in to_delete_dirs:
        try:
            shutil.rmtree(d)
            print(f"[dir ] 删除：{d}")
            deleted_dirs += 1
        except Exception as e:
            print(f"[dir ] 无法删除：{d}  -> {e}")

    print(f"[done] 目录 {root} 下删除文件 {deleted_files} 个、目录 {deleted_dirs} 个")
    return deleted_files, deleted_dirs

if __name__ == "__main__":
    total_f = total_d = 0
    for td in TARGET_DIRS:
        fcnt, dcnt = delete_items_with_substring(td)
        total_f += fcnt; total_d += dcnt
    print(f"\n[summary] 共删除文件 {total_f} 个、目录 {total_d} 个")




'''
# -*- coding: utf-8 -*-
import os
import random
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from typing import Tuple, Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# ========== 基础工具 ==========
def oneloop(element_list):
    pop_ele = element_list.pop()
    element_list.insert(0, pop_ele)
    return element_list

def decide_adj_with_bernoulli(transition_list, move_idx, adj_in_group, bernoulli_prob=0.8):
    if np.random.binomial(1, bernoulli_prob) == 0:
        return transition_list
    total_num = len(transition_list)
    grouped_adj = []
    n_group = total_num // adj_in_group
    assert n_group * adj_in_group == total_num
    for idx in range(n_group):
        group = [transition_list[k] for k in range(idx * adj_in_group, (idx + 1) * adj_in_group)]
        grouped_adj.append(group)
    if move_idx > 0:
        reorder_grouped_adj = oneloop(grouped_adj)
        for _ in range(move_idx - 1):
            reorder_grouped_adj = oneloop(reorder_grouped_adj)
    else:
        reorder_grouped_adj = grouped_adj
    new_transition_list = []
    for group in reorder_grouped_adj:
        new_transition_list += group
    return new_transition_list

# ========== 合成数据 + 训练输入文件 生成 ==========
def generate_synthetic_data(
    root_dir: str,
    T_length: int = 1200,
    graph_size: int = 20,
    K_of_graph: int = 3,
    rewire_rate: float = 0.5,
    num_series: int = 12,
    series_cluster: int = 3,
    cycle: int = 100,
    seed: int = 310058,
    bernoulli_prob: float = 0.8,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    生成马尔可夫随机游走驱动的多变量时间序列（num_series 条），每条序列取值映射到 [-1,1]。
    返回:
      final_data: shape [num_series, T_length] 的 float 数组（已归一化到 [-1,1]）
      adj_dict:   {'mat_0': A0, ..., 'mat_(num_series-1)': A_{N-1}} 每个为 [graph_size, graph_size]
    """
    np.random.seed(seed)
    random.seed(seed)
    os.makedirs(root_dir, exist_ok=True)

    # 生成 cluster 的核心小世界图
    transition_graph_core = [
        nx.adjacency_matrix(
            nx.connected_watts_strogatz_graph(graph_size, K_of_graph, rewire_rate)
        ).todense()
        for _ in range(series_cluster)
    ]

    cluster_member = num_series // series_cluster
    assert cluster_member * series_cluster == num_series

    transition_mat_list = []
    for idx in range(num_series):
        core_idx = idx // cluster_member
        core_adj = transition_graph_core[core_idx]
        new_adj = np.zeros((graph_size, graph_size))
        for n_r, row in enumerate(core_adj):
            zeros_num = (row == 0).sum()
            add_edges = np.random.binomial(1, 1 / (5 * graph_size), zeros_num)
            row = np.asarray(row).ravel()
            row[row == 0] = add_edges
            new_adj[n_r] = row
            new_adj[n_r][n_r] = 1
        # 行归一化成转移概率
        new_adj = new_adj / new_adj.sum(axis=1, keepdims=True)
        transition_mat_list.append(new_adj)

    # 保存所有 state-transition 矩阵
    adjacency_matrices = {f"mat_{idx}": transition_mat_list[idx] for idx in range(num_series)}
    np.savez(os.path.join(root_dir, "transition_matrices.npz"), **adjacency_matrices)

    # 多序列的随机游走
    mts_data = np.zeros((num_series, T_length)).astype(int)
    for ts in range(1, T_length - 1):
        circle_idx = ts // cycle
        current_transition_list = decide_adj_with_bernoulli(
            transition_mat_list, circle_idx, cluster_member, bernoulli_prob=bernoulli_prob
        )
        if ts % cycle == cycle - 1 and ts > 0:
            current_states = mts_data[:, ts]
            bins = np.bincount(current_states)
            popular_state = np.argmax(bins)
            mts_data[:, ts + 1] = popular_state
        else:
            for series_idx in range(num_series):
                latest_state = mts_data[series_idx][ts]
                transition_prob = current_transition_list[series_idx][int(latest_state)]
                randomwalk = np.random.choice(a=graph_size, size=1, p=transition_prob)[0]
                mts_data[series_idx][ts + 1] = randomwalk

    # 映射到 [-1,1]
    final_data = 2 * (mts_data / graph_size) - 1

    # 导出 CSV（可选，方便检查）
    mts_df = pd.DataFrame(final_data.T, columns=[f'Series_{i}' for i in range(num_series)])
    mts_df.to_csv(os.path.join(root_dir, "synthetic_time_series.csv"), index_label="Time_Step")

    # 可视化（热力图 + 片段曲线）
    data = np.load(os.path.join(root_dir, "transition_matrices.npz"))
    with PdfPages(os.path.join(root_dir, "transition_matrices_heatmaps.pdf")) as pdf:
        for key in data.files:
            matrix = data[key]
            plt.figure(figsize=(6, 5))
            plt.imshow(matrix, cmap='viridis', interpolation='nearest')
            plt.title(f'Heatmap of {key}')
            plt.colorbar(label='Transition Probability')
            plt.xlabel('To State')
            plt.ylabel('From State')
            pdf.savefig()
            plt.close()

    start_ts = max(0, T_length - 200)
    end_ts = T_length
    short_data = final_data[:, start_ts:end_ts]
    time_axis = np.arange(start_ts, end_ts)
    plot_num = min(5, num_series)
    fig, axes = plt.subplots(plot_num, 1, figsize=(12, 1.0 * plot_num), sharex=True)
    for i in range(plot_num):
        axes[i].plot(time_axis, short_data[i], label=f"Series {i}")
        axes[i].set_ylabel("Value")
        axes[i].set_title(f"Series {i}")
        axes[i].grid(True)
        axes[i].legend()
    axes[-1].set_xlabel("Time Step")
    plt.suptitle(f"Synthetic MTS (Steps {start_ts}~{end_ts - 1})", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(root_dir, "time_series_segment.pdf"))
    plt.close()

    with PdfPages(os.path.join(root_dir, "adjacency_and_correlation_matrices.pdf")) as pdf:
        for key in data.files:
            matrix = data[key]
            plt.figure(figsize=(6, 5))
            sns.heatmap(matrix, cmap='viridis')
            plt.title(f'Adjacency (Transition) Matrix - {key}')
            plt.xlabel('To State')
            plt.ylabel('From State')
            pdf.savefig()
            plt.close()
        window_size = 100
        step_size = 100
        num_windows = (T_length - window_size) // step_size + 1
        for win in range(num_windows):
            start = win * step_size
            end = start + window_size
            window_data = final_data[:, start:end]
            corr_matrix = np.corrcoef(window_data)
            plt.figure(figsize=(6, 5))
            sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title(f'Correlation Matrix (Steps {start}-{end-1})')
            plt.xlabel('Series Index')
            plt.ylabel('Series Index')
            pdf.savefig()
            plt.close()

    return final_data, adjacency_matrices

def _make_windows(
    data_arr: np.ndarray,  # [N, T]
    seq_len: int, pred_len: int,
    in_dim: int = 2, out_dim: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 [N, T] 的序列打成 Graph WaveNet/MTGNN 风格的窗口：
      x: [num_samples, seq_len, N, C_in]
      y: [num_samples, pred_len, N, C_out]
    这里默认 C_in=2：第1维是数值本身；第2维给一个简单的时间编码（归一化的 time-of-day 占位）
    """
    N, T = data_arr.shape
    max_start = T - (seq_len + pred_len)
    xs, ys = [], []
    # 简易时间编码（0~1）
    tcode = (np.arange(T) % 288) / 288.0  # 若不是交通数据无所谓：只是提供第二通道以配合 in_dim=2 的模型
    for s in range(max_start + 1):
        e = s + seq_len
        y_end = e + pred_len
        x_slice = data_arr[:, s:e]                  # [N, seq_len]
        y_slice = data_arr[:, e:y_end]              # [N, pred_len]
        x_feat0 = x_slice.T[:, :, None]             # [seq_len, N, 1]
        x_feat1 = tcode[s:e][:, None, None].repeat(N, axis=1)  # [seq_len, N, 1]
        x = np.concatenate([x_feat0, x_feat1], axis=2)         # [seq_len, N, 2]
        y = y_slice.T[:, :, None]                                # [pred_len, N, 1]
        xs.append(x)
        ys.append(y)
    x = np.stack(xs, axis=0)  # [num_samples, seq_len, N, 2]
    y = np.stack(ys, axis=0)  # [num_samples, pred_len, N, 1]
    return x.astype(np.float32), y.astype(np.float32)

def _split_train_val_test(num_samples: int, train_ratio=0.6, val_ratio=0.2):
    n_train = int(num_samples * train_ratio)
    n_val = int(num_samples * val_ratio)
    n_test = num_samples - n_train - n_val
    return n_train, n_val, n_test

def _save_npz_dataset(out_dir: str, x: np.ndarray, y: np.ndarray):
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "train.npz"), x=x['train'], y=y['train'])
    np.savez_compressed(os.path.join(out_dir, "val.npz"),   x=x['val'],   y=y['val'])
    np.savez_compressed(os.path.join(out_dir, "test.npz"),  x=x['test'],  y=y['test'])

def _export_adj_pkl(sensor_graph_dir: str, dataset_key: str, A_weighted: np.ndarray):
    """
    写出 Graph WaveNet/MTGNN 兼容的 adj_mx_*.pkl:
      内容为 (sensor_ids, sensor_id_to_ind, adj_mx)，其中 adj_mx 为 [N, N] 权重矩阵
    """
    os.makedirs(sensor_graph_dir, exist_ok=True)
    N = A_weighted.shape[0]
    sensor_ids = [str(i) for i in range(N)]
    sensor_id_to_ind = {sid: int(sid) for sid in sensor_ids}
    adj_mx = A_weighted.astype(np.float32)
    pkl_path = os.path.join(sensor_graph_dir, f"adj_mx_{dataset_key}.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump((sensor_ids, sensor_id_to_ind, adj_mx), f, protocol=pickle.HIGHEST_PROTOCOL)
    return pkl_path

def _build_weighted_adj_from_transition_mats(adj_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    把每条序列的 state 转移矩阵，汇总为一个 N×N 的“变量间相似度图”。
    这里用的是“转移矩阵的余弦相似度 + 对称化 + 归一化”的简单启发式。
    """
    mats = [v.reshape(-1) for _, v in sorted(adj_dict.items(), key=lambda x: int(x[0].split('_')[-1]))]
    M = np.stack(mats, axis=0)  # [N, graph_size*graph_size]
    # 余弦相似度
    eps = 1e-8
    M_norm = M / (np.linalg.norm(M, axis=1, keepdims=True) + eps)
    sim = M_norm @ M_norm.T     # [N, N] in [-1,1]
    sim = (sim + sim.T) / 2.0
    np.fill_diagonal(sim, 0.0)
    sim = np.clip(sim, 0, 1)    # 取非负部分当作权重
    # 稀疏化（可选）：保留每行 top-k
    k = max(3, int(0.1 * sim.shape[0]))
    A = np.zeros_like(sim)
    for i in range(sim.shape[0]):
        idx = np.argsort(sim[i])[::-1][:k]
        A[i, idx] = sim[i, idx]
    # 对称化 + 重归一化（行归一化，避免全 0）
    A = np.maximum(A, A.T)
    row_sum = A.sum(axis=1, keepdims=True) + 1e-8
    A = A / row_sum
    return A

def make_all_inputs_for_one_setting(
    setting_name: str,
    base_root: str = "./data/synthetic",
    seq_len: int = 12,
    pred_len: int = 12,
    params: dict = None
):
    """
    生成一套难度数据 + 训练用 npz + 传给 train.py 的邻接 pkl
    目录布局：
      data/
        SYNTHETIC_EASY/        <- (dataset 根路径，传给 --data)
           train.npz
           val.npz
           test.npz
           synthetic_time_series.csv
           transition_matrices.npz
           *.pdf
        sensor_graph/
           adj_mx_synthetic_easy.pkl  <- (传给 --adjdata)
    """
    if params is None:
        params = {}
    out_dir = os.path.join(base_root, setting_name)
    data_arr, adj_dict = generate_synthetic_data(out_dir, **params)

    # 打窗口 -> 划分
    x_all, y_all = _make_windows(data_arr, seq_len, pred_len, in_dim=2, out_dim=1)
    n_train, n_val, n_test = _split_train_val_test(x_all.shape[0], train_ratio=0.6, val_ratio=0.2)
    idx0 = 0
    splits_x = {
        'train': x_all[idx0: idx0 + n_train],
        'val':   x_all[idx0 + n_train: idx0 + n_train + n_val],
        'test':  x_all[idx0 + n_train + n_val:]
    }
    splits_y = {
        'train': y_all[idx0: idx0 + n_train],
        'val':   y_all[idx0 + n_train: idx0 + n_train + n_val],
        'test':  y_all[idx0 + n_train + n_val:]
    }

    # 保存 npz 到数据集根目录（让 --data 指到这里即可）
    dataset_key = f"synthetic_{setting_name}"
    dataset_root = os.path.join("./data", f"SYNTHETIC_{setting_name.upper()}")
    os.makedirs(dataset_root, exist_ok=True)
    _save_npz_dataset(dataset_root, splits_x, splits_y)

    # 把原始可视化与 csv 复制/重命名也放一份（可选，但便于核验）
    try:
        import shutil
        for fname in [
            "synthetic_time_series.csv",
            "transition_matrices.npz",
            "transition_matrices_heatmaps.pdf",
            "time_series_segment.pdf",
            "adjacency_and_correlation_matrices.pdf",
        ]:
            src = os.path.join(out_dir, fname)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(dataset_root, fname))
    except Exception:
        pass

    # 变量间图（N×N） -> 写出到 data/sensor_graph/adj_mx_*.pkl
    A_weighted = _build_weighted_adj_from_transition_mats(adj_dict)
    adj_pkl_path = _export_adj_pkl("./data/sensor_graph", dataset_key, A_weighted)

    # 记录 meta 信息，方便 train.py 里检查
    meta = {
        "seq_length": seq_len,
        "pred_length": pred_len,
        "num_nodes": int(data_arr.shape[0]),
        "dataset_root": dataset_root,
        "adj_pkl": adj_pkl_path,
        "data_csv": os.path.join(dataset_root, "synthetic_time_series.csv"),
    }
    pd.Series(meta).to_json(os.path.join(dataset_root, "meta.json"), force_ascii=False, indent=2)

    print(f"[OK] {setting_name}:")
    print(f"  --data    => {dataset_root}")
    print(f"  --adjdata => {adj_pkl_path}")
    print(f"  num_nodes => {data_arr.shape[0]}")
    print(f"  seq/pred  => {seq_len}/{pred_len}")
    print("")

if __name__ == "__main__":
    # 四套难度的默认参数（与你原来的设置一致）
    difficulty_settings = [
        {
            "name": "easy",
            "graph_size": 20,
            "series_cluster": 3,
            "cycle": 200,
            "num_series": 12,
            "K_of_graph": 3,
            "rewire_rate": 0.1,
            "T_length": 1200,
            "bernoulli_prob": 0.95,
        },
        {
            "name": "medium",
            "graph_size": 20,
            "series_cluster": 3,
            "cycle": 100,
            "num_series": 12,
            "K_of_graph": 3,
            "rewire_rate": 0.3,
            "T_length": 1200,
            "bernoulli_prob": 0.8,
        },
        {
            "name": "hard",
            "graph_size": 20,
            "series_cluster": 3,
            "cycle": 80,
            "num_series": 12,
            "K_of_graph": 3,
            "rewire_rate": 0.5,
            "T_length": 1200,
            "bernoulli_prob": 0.6,
        },
        {
            "name": "very_hard",
            "graph_size": 20,
            "series_cluster": 3,
            "cycle": 50,
            "num_series": 12,
            "K_of_graph": 3,
            "rewire_rate": 0.7,
            "T_length": 1200,
            "bernoulli_prob": 0.3,
        },
    ]

    # 你可以在这里统一改窗口长度（与 train.py 一致即可）
    SEQ_LEN = 12
    PRED_LEN = 12

    print("开始生成四套合成数据 + 训练输入文件 ...\n")
    for setting in difficulty_settings:
        make_all_inputs_for_one_setting(
            setting_name=setting["name"],
            base_root="./data/synthetic",
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN,
            params=dict(
                T_length=setting["T_length"],
                graph_size=setting["graph_size"],
                K_of_graph=setting["K_of_graph"],
                rewire_rate=setting["rewire_rate"],
                num_series=setting["num_series"],
                series_cluster=setting["series_cluster"],
                cycle=setting["cycle"],
                seed=310058,
                bernoulli_prob=setting["bernoulli_prob"],
            )
        )
    print("全部生成完成 ✅")
'''