# -*- coding: utf-8 -*-
"""
VAR Grid Search 实验
对四个合成数据集 (easy/medium/hard/very_hard) 运行 VAR
预测 horizon = [3,6,12,24]，保存每个数据集 + horizon 的结果到 CSV
"""

import os
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ================= 配置 =================
DATASETS = [
    "SYNTHETIC_EASY",
    "SYNTHETIC_MEDIUM",
    "SYNTHETIC_HARD",
    "SYNTHETIC_VERY_HARD"
]

HORIZONS = [3, 6, 12, 24]

# lag 搜索范围
LAG_VALUES = [1, 2, 3, 4, 5, 6, 12]

TRAIN_RATIO, VAL_RATIO = 0.6, 0.2
RESULT_CSV = "var_results.csv"


# ============== 工具函数 =================
def evaluate_var(data, lag, horizon, n_train, n_val, n_test):
    """
    data: shape [T, N]
    lag: VAR 滞后阶数
    horizon: 预测步长
    """
    train = data[:n_train]
    val   = data[n_train:n_train+n_val]
    test  = data[n_train+n_val:]

    # 如果 val/test 太短，直接返回 NaN
    if len(val) <= lag or len(test) <= horizon:
        return np.nan, np.nan, np.nan

    # ===== 验证集预测 =====
    history = np.vstack([train, val[:lag]])  # 起始窗口
    val_preds, val_true = [], []
    for t in range(lag, len(val) - horizon + 1):
        try:
            model = VAR(history)
            model_fit = model.fit(lag)
            forecast = model_fit.forecast(history[-lag:], steps=horizon)
            yhat = forecast[-1]  # horizon 步最后一个点
        except Exception:
            yhat = history[-1]
        val_preds.append(yhat)
        val_true.append(val[t + horizon - 1])
        history = np.vstack([history, val[t]])  # 滚动更新
    val_rmse = mean_squared_error(val_true, val_preds, squared=False)

    # ===== 测试集预测 =====
    history = np.vstack([train, val])
    test_preds, test_true = [], []
    for t in range(lag, len(test) - horizon + 1):
        try:
            model = VAR(history)
            model_fit = model.fit(lag)
            forecast = model_fit.forecast(history[-lag:], steps=horizon)
            yhat = forecast[-1]
        except Exception:
            yhat = history[-1]
        test_preds.append(yhat)
        test_true.append(test[t + horizon - 1])
        history = np.vstack([history, test[t]])
    test_rmse = mean_squared_error(test_true, test_preds, squared=False)
    test_mae  = mean_absolute_error(test_true, test_preds)

    return val_rmse, test_rmse, test_mae


def run_var_search(dataset_root, horizon):
    csv_path = os.path.join(dataset_root, "synthetic_time_series.csv")
    df = pd.read_csv(csv_path, index_col=0)
    values = df.values  # [T, N]

    T, N = values.shape
    n_train = int(T * TRAIN_RATIO)
    n_val   = int(T * VAL_RATIO)
    n_test  = T - n_train - n_val

    # 搜索最佳 lag
    best_lag, best_val_rmse = None, float("inf")
    for lag in LAG_VALUES:
        try:
            val_rmse, _, _ = evaluate_var(values, lag, horizon, n_train, n_val, n_test)
            if not np.isnan(val_rmse) and val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_lag = lag
        except Exception:
            continue

    # 用最佳 lag 计算测试集表现
    if best_lag is None:
        return np.nan, np.nan, None
    else:
        _, test_rmse, test_mae = evaluate_var(values, best_lag, horizon, n_train, n_val, n_test)
        return test_rmse, test_mae, best_lag


# ============== 主流程 =================
if __name__ == "__main__":
    records = []

    for dataset in DATASETS:
        dataset_root = f"./data/{dataset}"
        for horizon in HORIZONS:
            rmse, mae, lag = run_var_search(dataset_root, horizon)
            records.append({
                "dataset": dataset,
                "horizon": horizon,
                "rmse": rmse,
                "mae": mae,
                "best_lag": lag
            })
            print(f"{dataset} | horizon={horizon}: RMSE={rmse:.4f}, MAE={mae:.4f}, lag={lag}")

        print(f"=== {dataset} 完成 ✅ ===")

    # 保存结果
    df_result = pd.DataFrame(records)
    df_result.to_csv(RESULT_CSV, index=False)
    print(f"\n结果已保存到 {RESULT_CSV}")




# python -u run_var_gridsearch.py
