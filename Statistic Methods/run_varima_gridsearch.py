# -*- coding: utf-8 -*-
"""
VARIMA (通过 VARMAX 实现) Grid Search 实验
使用 method="powell" + maxiter=20 避免卡死
默认只取前 3 个变量测试（想跑全部变量可注释掉相应代码）
"""

import os
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.varmax import VARMAX
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

# 搜索范围（小范围先跑通，确认后可放大）
P_VALUES = [1, 2]
D_VALUES = [0, 1]
Q_VALUES = [0]

TRAIN_RATIO, VAL_RATIO = 0.6, 0.2
RESULT_CSV = "varima_results.csv"


# ============== 工具函数 =================
def difference(data, d):
    """简单差分 (d=0 返回原始数据)"""
    if d == 0:
        return data
    diffed = data.copy()
    for _ in range(d):
        diffed = np.diff(diffed, axis=0)
    return diffed


def evaluate_varima(data, order, d, horizon, n_train, n_val, n_test):
    """VARIMA(p,d,q) 预测"""
    diffed = difference(data, d)
    if diffed.shape[0] < (n_train + n_val + n_test):
        return np.nan, np.nan, np.nan

    train = diffed[:n_train]
    val   = diffed[n_train:n_train+n_val]
    test  = diffed[n_train+n_val:]

    if len(val) <= max(order) or len(test) <= horizon:
        return np.nan, np.nan, np.nan

    # ===== 验证集预测 =====
    history = train.copy()
    val_preds, val_true = [], []
    for t in range(len(val) - horizon + 1):
        try:
            model = VARMAX(history, order=order, enforce_stationarity=False)
            model_fit = model.fit(disp=False, maxiter=20, method="powell")
            forecast = model_fit.forecast(steps=horizon)
            yhat = forecast.iloc[-1].values
        except Exception:
            yhat = history[-1]
        val_preds.append(yhat)
        val_true.append(val[t + horizon - 1])
        history = np.vstack([history, val[t]])
    val_rmse = mean_squared_error(val_true, val_preds, squared=False)

    # ===== 测试集预测 =====
    history = np.vstack([train, val])
    test_preds, test_true = [], []
    for t in range(len(test) - horizon + 1):
        try:
            model = VARMAX(history, order=order, enforce_stationarity=False)
            model_fit = model.fit(disp=False, maxiter=20, method="powell")
            forecast = model_fit.forecast(steps=horizon)
            yhat = forecast.iloc[-1].values
        except Exception:
            yhat = history[-1]
        test_preds.append(yhat)
        test_true.append(test[t + horizon - 1])
        history = np.vstack([history, test[t]])
    test_rmse = mean_squared_error(test_true, test_preds, squared=False)
    test_mae  = mean_absolute_error(test_true, test_preds)

    return val_rmse, test_rmse, test_mae


def run_varima_search(dataset_root, horizon):
    csv_path = os.path.join(dataset_root, "synthetic_time_series.csv")
    df = pd.read_csv(csv_path, index_col=0)

    # ⚠️ 默认只取前 3 个变量测试，确认无误后可注释掉
    df = df.iloc[:, :3]

    values = df.values  # [T, N]

    T, N = values.shape
    n_train = int(T * TRAIN_RATIO)
    n_val   = int(T * VAL_RATIO)
    n_test  = T - n_train - n_val

    best_order, best_d, best_val_rmse = None, None, float("inf")

    combos = [(p,d,q) for p in P_VALUES for d in D_VALUES for q in Q_VALUES]

    for (p,d,q) in tqdm(combos, desc=f"{os.path.basename(dataset_root)} horizon={horizon}"):
        order = (p, q)
        print(f"  >> Trying VARIMA order={order}, d={d}, horizon={horizon}")
        try:
            val_rmse, _, _ = evaluate_varima(values, order, d, horizon, n_train, n_val, n_test)
            if not np.isnan(val_rmse) and val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_order = order
                best_d = d
        except Exception as e:
            print(f"     Failed: order={order}, d={d}, error={str(e)}")
            continue

    if best_order is None:
        return np.nan, np.nan, None

    _, test_rmse, test_mae = evaluate_varima(values, best_order, best_d, horizon, n_train, n_val, n_test)
    return test_rmse, test_mae, (best_order, best_d)


# ============== 主流程 =================
if __name__ == "__main__":
    records = []

    for dataset in DATASETS:
        dataset_root = f"./data/{dataset}"
        for horizon in HORIZONS:
            rmse, mae, params = run_varima_search(dataset_root, horizon)
            records.append({
                "dataset": dataset,
                "horizon": horizon,
                "rmse": rmse,
                "mae": mae,
                "best_params": params
            })
            print(f"{dataset} | horizon={horizon}: RMSE={rmse:.4f}, MAE={mae:.4f}, params={params}")

        print(f"=== {dataset} 完成 ✅ ===")

    df_result = pd.DataFrame(records)
    df_result.to_csv(RESULT_CSV, index=False)
    print(f"\n结果已保存到 {RESULT_CSV}")


# python -u run_varima_gridsearch.py
