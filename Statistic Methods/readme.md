# Classical Time Series Baselines — Grid Search (ARIMA / VAR / VARIMA)

A small toolkit to run **grid search** for classical time-series baselines on **four synthetic datasets**, and to save the best hyperparameters and test metrics for each `(dataset, horizon)` into CSV files.

Included scripts:

- `run_arima_gridsearch.py` — univariate **ARIMA** (grid search per variable/series)
- `run_var_gridsearch.py` — multivariate **VAR**
- `run_varima_gridsearch.py` — multivariate **VARIMA** (implemented via `statsmodels` **VARMAX**)

---

## Data layout and format

By default, the scripts expect data at:

```
./data/<DATASET_NAME>/synthetic_time_series.csv
```

Default dataset names:

- `SYNTHETIC_EASY`
- `SYNTHETIC_MEDIUM`
- `SYNTHETIC_HARD`
- `SYNTHETIC_VERY_HARD`

`synthetic_time_series.csv` must follow:

- rows = time steps `T`
- columns = variables `N`
- **the first column is an index** (read as index and effectively ignored), remaining columns are numeric variables

Conceptual example:

| index | x1 | x2 | x3 |
|---:|---:|---:|---:|
| 0 | 1.2 | 0.1 | 5.0 |
| 1 | 1.1 | 0.2 | 5.1 |
| … | … | … | … |

---

## Installation

Recommended: use a virtual environment (venv / conda).

Minimal dependencies (matching the imports in the scripts):

- Python 3.9+
- `pandas`
- `numpy`
- `statsmodels`
- `scikit-learn`
- `tqdm`

Install:

```bash
pip install -U pandas numpy statsmodels scikit-learn tqdm
```

---

## Shared experimental setup (all scripts)

- Split: `train/val/test = 0.6 / 0.2 / 0.2`
- Forecast horizons: `[3, 6, 12, 24]`
- Evaluation: rolling / walk-forward `horizon-step` forecasting; for each window, the prediction is compared against the **last step** of the `horizon`
- Metrics:
  - RMSE (used for validation model selection; also reported on test)
  - MAE (reported on test)

---

## Usage

### 1) ARIMA grid search (univariate, per series)

```bash
python -u run_arima_gridsearch.py
```

Output: `arima_results.csv`

Default search space:

- `p ∈ [0,1,2,3]`
- `d ∈ [0,1]`
- `q ∈ [0,1,2,3]`

Notes / safeguards:

- **Constant series are skipped** (results become NaN)
- On fit errors, it falls back to the **last observed value** as the prediction
- `maxiter=50` (to avoid excessively long fits)

---

### 2) VAR grid search (multivariate)

```bash
python -u run_var_gridsearch.py
```

Output: `var_results.csv`

Default lag search space:

- `lag ∈ [1,2,3,4,5,6,12]`

---

### 3) VARIMA grid search (via VARMAX)

```bash
python -u run_varima_gridsearch.py
```

Output: `varima_results.csv`

Default search space (kept small to run reliably):

- `p ∈ [1,2]`
- `d ∈ [0,1]` (implemented by **simple differencing** in the script)
- `q ∈ [0]`  
- In `VARMAX`, the model uses `order=(p, q)` and applies differencing according to `d`

⚠️ Important default behavior:

- To prevent `VARMAX` from becoming too slow in higher dimensions:  
  the script **only uses the first 3 variables by default** (`df = df.iloc[:, :3]`).  
  Comment out that line to use all variables.
- Fit settings: `method="powell"`, `maxiter=20` (to avoid getting stuck)

---

## Outputs (CSV columns)

### `arima_results.csv`

| Column | Meaning |
|---|---|
| `dataset` | dataset name |
| `horizon` | forecast horizon |
| `rmse` | mean **test** RMSE across variables (`nanmean`) |
| `mae` | mean **test** MAE across variables (`nanmean`) |
| `best_orders_per_series` | best `(p,d,q)` per variable (constant/failures are `None`) |

### `var_results.csv`

| Column | Meaning |
|---|---|
| `dataset` | dataset name |
| `horizon` | forecast horizon |
| `rmse` | test RMSE |
| `mae` | test MAE |
| `best_lag` | best VAR lag |

### `varima_results.csv`

| Column | Meaning |
|---|---|
| `dataset` | dataset name |
| `horizon` | forecast horizon |
| `rmse` | test RMSE |
| `mae` | test MAE |
| `best_params` | best parameters: `((p, q), d)` |

---

## Tips & troubleshooting

- **It runs too slowly**
  - ARIMA: shrink `P_VALUES / Q_VALUES` or reduce the number of variables
  - VAR: shrink `LAG_VALUES`
  - VARIMA/VARMAX: reduce dimensionality first (runtime grows quickly with #variables); keep `maxiter` small
- **NaNs in results**
  - constant series, too-short validation/test slices, or repeated fit failures can cause NaNs
- **Use your own dataset**
  - place your CSV at `./data/<NAME>/synthetic_time_series.csv`
  - update `DATASETS = [...]` in the scripts accordingly

---

## Reproducible run order

```bash
python -u run_arima_gridsearch.py
python -u run_var_gridsearch.py
python -u run_varima_gridsearch.py
```



- `arima_results.csv`
- `var_results.csv`
- `varima_results.csv`



