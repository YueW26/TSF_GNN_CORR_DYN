# -*- coding: utf-8 -*-
"""
ARIMA Grid Search 实验 + 安全保护
"""

import os
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
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

# (p,d,q) 搜索范围
P_VALUES = [0, 1, 2, 3]
D_VALUES = [0, 1]
Q_VALUES = [0, 1, 2, 3]

TRAIN_RATIO, VAL_RATIO = 0.6, 0.2
RESULT_CSV = "arima_results.csv"


# ============== 工具函数 =================
def evaluate_arima(series, order, horizon, n_train, n_val, n_test):
    """
    horizon-step 预测，滚动预测
    """
    train = series[:n_train]
    val   = series[n_train:n_train+n_val]
    test  = series[n_train+n_val:]

    # 如果 val 或 test 太短，直接跳过
    if len(val) < horizon or len(test) < horizon:
        return np.nan, np.nan, np.nan

    # 验证集预测
    history = list(train)
    val_preds = []
    for t in range(len(val) - horizon + 1):
        try:
            model = ARIMA(history, order=order)
            model_fit = model.fit(method_kwargs={"maxiter": 50}, disp=0)
            yhat = model_fit.forecast(steps=horizon)[-1]  # 取 horizon 步的最后一个点
        except Exception:
            yhat = history[-1]
        val_preds.append(yhat)
        history.append(val[t + horizon - 1])
    val_true = val[horizon-1:]
    val_rmse = mean_squared_error(val_true, val_preds, squared=False)

    # 测试集预测
    test_preds = []
    history = list(train) + list(val)
    for t in range(len(test) - horizon + 1):
        try:
            model = ARIMA(history, order=order)
            model_fit = model.fit(method_kwargs={"maxiter": 50}, disp=0)
            yhat = model_fit.forecast(steps=horizon)[-1]
        except Exception:
            yhat = history[-1]
        test_preds.append(yhat)
        history.append(test[t + horizon - 1])
    test_true = test[horizon-1:]
    test_rmse = mean_squared_error(test_true, test_preds, squared=False)
    test_mae  = mean_absolute_error(test_true, test_preds)

    return val_rmse, test_rmse, test_mae


def run_arima_search(dataset_root, horizon):
    csv_path = os.path.join(dataset_root, "synthetic_time_series.csv")
    df = pd.read_csv(csv_path, index_col=0)
    values = df.values  # [T, N]

    T, N = values.shape
    n_train = int(T * TRAIN_RATIO)
    n_val   = int(T * VAL_RATIO)
    n_test  = T - n_train - n_val

    best_orders = []
    test_results = []

    for i in tqdm(range(N), desc=f"{os.path.basename(dataset_root)} horizon={horizon}", leave=False):
        series = values[:, i]

        # 跳过常数序列
        if np.all(series == series[0]):
            best_orders.append(None)
            test_results.append((np.nan, np.nan))
            continue

        # 搜索最优 (p,d,q)
        best_order, best_val_rmse = None, float("inf")
        for p in P_VALUES:
            for d in D_VALUES:
                for q in Q_VALUES:
                    order = (p,d,q)
                    try:
                        val_rmse, _, _ = evaluate_arima(series, order, horizon, n_train, n_val, n_test)
                        if not np.isnan(val_rmse) and val_rmse < best_val_rmse:
                            best_val_rmse = val_rmse
                            best_order = order
                    except Exception:
                        continue

        # 用最佳参数计算测试集表现
        if best_order is None:
            best_orders.append(None)
            test_results.append((np.nan, np.nan))
        else:
            _, test_rmse, test_mae = evaluate_arima(series, best_order, horizon, n_train, n_val, n_test)
            best_orders.append(best_order)
            test_results.append((test_rmse, test_mae))

    avg_rmse = np.nanmean([r for r,_ in test_results])
    avg_mae  = np.nanmean([m for _,m in test_results])
    return avg_rmse, avg_mae, best_orders


# ============== 主流程 =================
if __name__ == "__main__":
    records = []

    for dataset in DATASETS:
        dataset_root = f"./data/{dataset}"
        for horizon in HORIZONS:
            rmse, mae, orders = run_arima_search(dataset_root, horizon)
            records.append({
                "dataset": dataset,
                "horizon": horizon,
                "rmse": rmse,
                "mae": mae,
                "best_orders_per_series": orders
            })
            print(f"{dataset} | horizon={horizon}: RMSE={rmse:.4f}, MAE={mae:.4f}")

        print(f"=== {dataset} 完成 ✅ ===")

    # 保存结果
    df_result = pd.DataFrame(records)
    df_result.to_csv(RESULT_CSV, index=False)
    print(f"\n结果已保存到 {RESULT_CSV}")


# python -u run_arima_gridsearch.py
