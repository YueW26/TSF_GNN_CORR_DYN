# prepare_real_datasets.py
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import pickle

def load_real_data(csv_path: str) -> np.ndarray:
    """
    读取真实数据 (solar/electricity)，返回 [N, T] numpy 数组
    """
    df = pd.read_csv(csv_path)
    # 若第一列是时间戳则去掉
    if not np.issubdtype(df.dtypes[0], np.number):
        df = df.iloc[:, 1:]
    data = df.values.T  # 转置: [N, T]
    # 归一化到 [-1, 1]
    data = (data - data.min()) / (data.max() - data.min())
    data = 2 * data - 1
    return data.astype(np.float32)


def build_similarity_graph(data_arr: np.ndarray, top_k: int = 10) -> np.ndarray:
    """
    根据变量间相关性构造邻接矩阵
    data_arr: [N, T]
    返回: [N, N]
    """
    corr = np.corrcoef(data_arr)
    np.fill_diagonal(corr, 0.0)
    N = data_arr.shape[0]
    A = np.zeros_like(corr)
    for i in range(N):
        idx = np.argsort(corr[i])[::-1][:top_k]
        A[i, idx] = corr[i, idx]
    A = (A + A.T) / 2
    A = A / (A.sum(axis=1, keepdims=True) + 1e-8)
    return A.astype(np.float32)


def _make_windows(data_arr: np.ndarray, seq_len: int, pred_len: int) -> tuple:
    """
    将 [N, T] 的序列打成 Graph WaveNet/MTGNN 风格的窗口：
      x: [num_samples, seq_len, N, 2]
      y: [num_samples, pred_len, N, 1]
    """
    N, T = data_arr.shape
    max_start = T - (seq_len + pred_len)
    xs, ys = [], []
    tcode = (np.arange(T) % 288) / 288.0  # 简单时间编码
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


def _save_npz_dataset(out_dir: str, x: dict, y: dict):
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "train.npz"), x=x['train'], y=y['train'])
    np.savez_compressed(os.path.join(out_dir, "val.npz"),   x=x['val'],   y=y['val'])
    np.savez_compressed(os.path.join(out_dir, "test.npz"),  x=x['test'],  y=y['test'])


def _export_adj_pkl(sensor_graph_dir: str, dataset_key: str, A_weighted: np.ndarray):
    """
    写出 Graph WaveNet/MTGNN 兼容的 adj_mx_*.pkl
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


def make_real_dataset(csv_path: str, dataset_key: str, seq_len: int, pred_len: int):
    data_arr = load_real_data(csv_path)
    x_all, y_all = _make_windows(data_arr, seq_len, pred_len)

    n_train, n_val, n_test = _split_train_val_test(x_all.shape[0])
    splits_x = {
        'train': x_all[:n_train],
        'val':   x_all[n_train:n_train+n_val],
        'test':  x_all[n_train+n_val:]
    }
    splits_y = {
        'train': y_all[:n_train],
        'val':   y_all[n_train:n_train+n_val],
        'test':  y_all[n_train+n_val:]
    }

    dataset_root = os.path.join("./data", dataset_key.upper())
    os.makedirs(dataset_root, exist_ok=True)
    _save_npz_dataset(dataset_root, splits_x, splits_y)

    # adjacency
    A = build_similarity_graph(data_arr)
    adj_pkl_path = _export_adj_pkl("./data/sensor_graph", dataset_key, A)

    print(f"[OK] {dataset_key}:")
    print(f"  --data    => {dataset_root}")
    print(f"  --adjdata => {adj_pkl_path}")
    print(f"  num_nodes => {data_arr.shape[0]}")
    print(f"  seq/pred  => {seq_len}/{pred_len}")
    print("")


if __name__ == "__main__":
    SEQ_LEN = 12
    PRED_LEN = 12

    make_real_dataset(
        "/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/GNN/Graph-WaveNet-master-origin/data/solar.csv",
        "solar", SEQ_LEN, PRED_LEN
    )
    make_real_dataset(
        "/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/GNN/Graph-WaveNet-master-origin/data/electricity.csv",
        "electricity", SEQ_LEN, PRED_LEN
    )

    print("Solar/Electricity 数据集已转换完成 ✅")
