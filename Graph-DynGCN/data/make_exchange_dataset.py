import numpy as np
import os
import argparse
import pickle

def load_exchange_rate(path):
    data = np.loadtxt(path, delimiter=",")
    print(f"[INFO] Loaded exchange_rate: shape={data.shape} (T, N)")
    return data

def generate_dataset(data, seq_length=12, pred_length=12):
    T, N = data.shape
    num_samples = T - seq_length - pred_length + 1

    # 修改：把最后一维从 1 -> 2（多一个通道）
    x = np.zeros((num_samples, seq_length, N, 2))
    y = np.zeros((num_samples, pred_length, N, 1))

    for i in range(num_samples):
        x_slice = data[i:i+seq_length]                    # (seq_length, N)
        y_slice = data[i+seq_length:i+seq_length+pred_length]  # (pred_length, N)

        # 通道1：原始数值
        x1 = np.expand_dims(x_slice, axis=-1)  # (seq_length, N, 1)

        # 通道2：直接复制原始数值（你也可以替换成时间特征等）
        x2 = np.expand_dims(x_slice, axis=-1)  # (seq_length, N, 1)

        # 拼接成两通道
        x[i] = np.concatenate([x1, x2], axis=-1)  # (seq_length, N, 2)

        # y 还是单通道
        y[i] = np.expand_dims(y_slice, axis=-1)   # (pred_length, N, 1)

    print(f"[INFO] Generated dataset: x={x.shape}, y={y.shape}")
    return x, y

def split_and_save(x, y, out_dir, train_ratio=0.6, val_ratio=0.2):
    os.makedirs(out_dir, exist_ok=True)
    num_samples = x.shape[0]
    n_train = int(num_samples * train_ratio)
    n_val = int(num_samples * val_ratio)
    n_test = num_samples - n_train - n_val

    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:n_train+n_val], y[n_train:n_train+n_val]
    x_test, y_test = x[n_train+n_val:], y[n_train+n_val:]

    np.savez_compressed(os.path.join(out_dir, "train.npz"), x=x_train, y=y_train)
    np.savez_compressed(os.path.join(out_dir, "val.npz"),   x=x_val,   y=y_val)
    np.savez_compressed(os.path.join(out_dir, "test.npz"),  x=x_test,  y=y_test)

    print(f"[OK] train={x_train.shape}, val={x_val.shape}, test={x_test.shape}")

def build_identity_adj(num_nodes, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 节点 ID 用字符串 "0", "1", ..., "N-1"
    sensor_ids = [str(i) for i in range(num_nodes)]
    sensor_id_to_ind = {str(i): i for i in range(num_nodes)}

    # 邻接矩阵用单位矩阵（模型会学习自适应的 adj）
    adj_mx = np.eye(num_nodes)

    with open(out_path, "wb") as f:
        pickle.dump((sensor_ids, sensor_id_to_ind, adj_mx), f)

    print(f"[OK] Saved adj triple to {out_path}, shape={adj_mx.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="exchange_rate.txt")
    parser.add_argument("--out_dir", type=str, default="EXCHANGE")
    parser.add_argument("--seq_length", type=int, default=12)
    parser.add_argument("--pred_length", type=int, default=12)
    args = parser.parse_args()

    data = load_exchange_rate(args.data_path)
    x, y = generate_dataset(data, args.seq_length, args.pred_length)
    split_and_save(x, y, args.out_dir)
    build_identity_adj(data.shape[1], os.path.join("sensor_graph", "adj_mx_exchange_rate.pkl"))
