# make_pems_dataset.py
# -*- coding: utf-8 -*-
import os
import h5py
import numpy as np
import pickle


def load_pems_data(h5_path: str) -> np.ndarray:
    """
    从 pems-bay.h5 读取时间序列数据
    返回 [N, T] numpy 数组
    """
    with h5py.File(h5_path, 'r') as f:
        data = f["speed/block0_values"][:]  # shape [T, N]
    print(f"[INFO] 原始数据 shape: {data.shape} (T, N)")
    return data.T.astype(np.float32)  # 转置成 [N, T]


def make_windows(data_arr: np.ndarray, seq_len: int, pred_len: int):
    """
    将 [N, T] 序列切成窗口:
      x: [num_samples, seq_len, N, 2]
      y: [num_samples, pred_len, N, 1]
    """
    N, T = data_arr.shape
    max_start = T - (seq_len + pred_len)
    xs, ys = [], []
    # PEMS-BAY: 5分钟一个点，一天288步
    tcode = (np.arange(T) % 288) / 288.0  # 时间编码
    for s in range(max_start + 1):
        e = s + seq_len
        y_end = e + pred_len
        x_slice = data_arr[:, s:e]                 # [N, seq_len]
        y_slice = data_arr[:, e:y_end]             # [N, pred_len]
        x_feat0 = x_slice.T[:, :, None]            # [seq_len, N, 1]
        x_feat1 = tcode[s:e][:, None, None].repeat(N, axis=1)  # [seq_len, N, 1]
        x = np.concatenate([x_feat0, x_feat1], axis=2)         # [seq_len, N, 2]
        y = y_slice.T[:, :, None]                               # [pred_len, N, 1]
        xs.append(x)
        ys.append(y)
    x = np.stack(xs, axis=0)
    y = np.stack(ys, axis=0)
    return x.astype(np.float32), y.astype(np.float32)


def split_train_val_test(num_samples: int, train_ratio=0.7, val_ratio=0.2):
    n_train = int(num_samples * train_ratio)
    n_val = int(num_samples * val_ratio)
    n_test = num_samples - n_train - n_val
    return n_train, n_val, n_test


def save_npz_dataset(out_dir: str, x: dict, y: dict):
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "train.npz"), x=x['train'], y=y['train'])
    np.savez_compressed(os.path.join(out_dir, "val.npz"),   x=x['val'],   y=y['val'])
    np.savez_compressed(os.path.join(out_dir, "test.npz"),  x=x['test'],  y=y['test'])


def make_pems_dataset(h5_path: str, adj_pkl_path: str,
                      dataset_key: str, seq_len: int, pred_len: int):
    # === 读取数据 ===
    data_arr = load_pems_data(h5_path)  # [N, T]
    print(f"[INFO] 转置后数据 shape: {data_arr.shape} (N, T)")

    # === 窗口化 ===
    x_all, y_all = make_windows(data_arr, seq_len, pred_len)
    print(f"[INFO] 窗口化结果: x={x_all.shape}, y={y_all.shape}")

    # === 数据划分 ===
    n_train, n_val, n_test = split_train_val_test(x_all.shape[0])
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

    # === 保存 npz ===
    dataset_root = os.path.join("./data", dataset_key.upper())
    os.makedirs(dataset_root, exist_ok=True)
    save_npz_dataset(dataset_root, splits_x, splits_y)

    print(f"[OK] {dataset_key}:")
    print(f"  --data    => {os.path.abspath(dataset_root)}")
    print(f"  --adjdata => {os.path.abspath(adj_pkl_path)}")
    print(f"  num_nodes => {data_arr.shape[0]}")
    print(f"  seq/pred  => {seq_len}/{pred_len}")


if __name__ == "__main__":
    SEQ_LEN = 12
    PRED_LEN = 12
    make_pems_dataset(
        "/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/GNN/Graph-WaveNet-master-origin/data/pems-bay.h5",
        "/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/GNN/Graph-WaveNet-master-origin/data/adj_mx_bay.pkl",
        "pemsbay", SEQ_LEN, PRED_LEN
    )
