import os
import numpy as np

def check_npz_file(path):
    print(f"\n>>> Checking {path}")
    data = np.load(path, allow_pickle=True)
    info = {}
    for key in data.files:
        arr = data[key]
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
        info[key] = arr.shape
    return info

if __name__ == "__main__":
    # 项目根目录
    data_dir = "data/SYNTHETIC_EASY"
    files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]

    if not files:
        print(f"No .npz files found in {data_dir}")
    else:
        for f in files:
            info = check_npz_file(os.path.join(data_dir, f))
            # 推断 seq_len / pred_len
            if "x" in info and "y" in info:
                seq_len = info["x"][1] if len(info["x"]) > 1 else None
                pred_len = info["y"][1] if len(info["y"]) > 1 else None
                num_nodes = info["x"][2] if len(info["x"]) > 2 else None
                print(f"  >>> 推断结果: seq_len={seq_len}, pred_len={pred_len}, 节点数={num_nodes}")
                if seq_len == 24 and pred_len == 24:
                    print("  ⚠️ 当前数据是固定 24 步输入和 24 步预测")
                    print("     如果你只想缩短预测步长，可以直接裁剪 y；")
                    print("     如果你要缩短输入步长，建议重新生成数据。")
