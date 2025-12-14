# DynGCN: 

DynGCN is a **spatiotemporal forecasting model** proposed in our work. It combines a **GraphWaveNet-like temporal backbone** (dilated gated temporal convolutions) with **DynGCN-specific spatial propagation modules**, including **dual-graph propagation** and **power-weighted mixing**. This repo also provides a **reproducible experiment runner** (`run_experiments_ab.sh`) for large-scale sweeps over datasets and hyperparameters.

---

## What’s included

### Model (`gwnet` / DynGCN core)
The model class in the code is named `gwnet` (legacy naming), but it **implements DynGCN**. It consists of:

- **Temporal backbone**: dilated causal convolutions with gated activations, residual/skip connections.
- **Spatial module (pluggable)**:
  - **Diffusion GCN (baseline)**: multi-step neighborhood diffusion.
  - **PowerLaw GCN**: uses matrix powers \(A^k\) with learnable hop coefficients.
  - **Chebyshev Convolution**: Chebyshev polynomial approximation on a normalized Laplacian.
  - **MixPropDual (DynGCN)**: **dual-graph propagation** using:
    - `adj_1`: normalized from a base adjacency (optionally from Laplacian),
    - `adj_2`: learned from node embeddings (`src_emb`, `dst_emb`).
  - **PowerMixDual (DynGCN)**: dual-graph propagation + **power-weighted injection** per propagation step.

### Experiment runner
- `run_experiments_ab.sh`: selects experiment groups via `EXP_ID`, sweeps grids, and launches training.
- `_wandb_proxy.py`: wrapper that starts a wandb run and executes the training command (expected in repo).
- `train.py`: training entrypoint (expected in repo).

---

## Repository expectations

The sweep script assumes these files exist at repo root:

- `train.py` — training script (loads data, builds DynGCN, trains/evaluates, writes metrics)
- `_wandb_proxy.py` — wandb wrapper used by the runner
- `run_experiments_ab.sh` — experiment/sweep runner
- `data/` — datasets (one folder per dataset)

---

## Installation

The exact dependencies depend on your `train.py`, but a typical setup is:

- Python 3.9+
- PyTorch (+ CUDA if using GPU)
- numpy / pandas
- wandb (optional)

Example:

```bash
pip install -U torch numpy pandas wandb
```

---

## Data layout

The runner passes dataset folders (e.g., `data/EXCHANGE`) via `--data`:

```bash
python train.py --data data/EXCHANGE ...
```

So `train.py` is responsible for interpreting the dataset folder content (e.g., `.npz`, `.pkl`, `.npy`, CSV, etc.).  
Place your datasets under `data/<DATASET_NAME>/...`.

---

## Model overview

### Temporal backbone
DynGCN uses:
- gated temporal convolutions (`tanh(filter) * sigmoid(gate)`),
- exponentially increasing dilations per layer,
- residual + skip connections,
- two end conv layers to output `out_dim` time steps.

Key knobs:
- `blocks`, `layers` → control receptive field
- `residual_channels`, `dilation_channels`, `skip_channels`, `end_channels`

### Graph supports and adaptive adjacency
DynGCN takes a list of support matrices `supports` (each `[N, N]`). If `gcn_bool=True` and `addaptadj=True`, it also learns an **adaptive adjacency**:

\[
A_{adp} = softmax(ReLU(nodevec1 \cdot nodevec2))
\]

All supports + adaptive adjacency are collected at runtime and used by the selected spatial module.

### Diagonal policy (`diag_mode`)
Controls whether the diagonal/self-loop is kept:

- `self_and_neighbor` (default): keep diagonal
- `neighbor`: remove diagonal (sets diagonal to zero)

This affects the diffusion/PowerLaw GCN, Chebyshev, MixPropDual, and PowerMixDual modules.

---

## Spatial variants (how to choose)

In the model code, exactly **one** spatial module is active at a time (per layer), selected in this priority order:

1. `use_cheby=True` → Chebyshev (`ChebConv`)
2. `use_mixprop=True` → MixPropDual (DynGCN)
3. `use_powermix=True` → PowerMixDual (DynGCN)
4. else → diffusion GCN baseline (`gcn`, optionally with `use_power=True` for PowerLaw)

### Diffusion GCN baseline
- multi-step diffusion using repeated `nconv(x, A)`,
- concatenates diffusion outputs and projects with a 1×1 conv.

### PowerLaw GCN
- uses **matrix powers** `A, A^2, ..., A^K`,
- each hop has a learnable coefficient `power_coef[k]`.

> `power_init` exists in the DynGCN constructor; applying specific initialization strategies (plain/decay/softmax) should be done in your model-building code (e.g., inside `train.py`).

### Chebyshev (spectral)
- builds normalized Laplacian `L`, rescales to `L_tilde`,
- computes Chebyshev polynomials up to order `K`,
- mixes them with learnable weights `alpha`.

### MixPropDual (DynGCN)
- builds `adj_1` from base adjacency (or Laplacian if enabled),
- builds `adj_2` from node embeddings (`src_emb`, `dst_emb`),
- propagates through both graphs and sums the results.

For analysis/debugging, the module stores:
- `self.adj_1` and `self.adj_2` (detached) each forward pass.

### PowerMixDual (DynGCN)
- dual-graph propagation as above,
- additionally uses `power_coef[k]` to weight injected signals at step `k`.

For analysis/debugging, it stores:
- `self.adj_1` and `self.adj_2` (detached) each forward pass.

---

## Training entrypoint (`train.py`)

The runner expects `train.py` to support (at least) these CLI flags:

```bash
--data <path> --device <cuda:0/cpu> --batch_size <int> --epochs <int>
--seq_length <int> --pred_length <int>
--learning_rate <float> --dropout <float> --nhid <int>
--weight_decay <float> --print_every <int>
--gcn_bool --randomadj --adjtype <doubletransition/...>
--blocks <int> --layers <int>
--addaptadj   # optional (controlled by DISABLE_ADAPTADJ)
```

and to map environment-variable switches (below) to DynGCN configuration.

---

## Running experiments

### One-liners

Baseline sweep:
```bash
EXP_ID=1 DEVICE=cuda:0 bash run_experiments_ab.sh
```

PowerLaw ablation:
```bash
EXP_ID=2 DEVICE=cuda:0 bash run_experiments_ab.sh
```

DynGCN PowerMixDual sweep:
```bash
EXP_ID=6 DEVICE=cuda:0 bash run_experiments_ab.sh
```

Run everything (can be very large):
```bash
EXP_ID=0 DEVICE=cuda:0 bash run_experiments_ab.sh
```

---

## Experiment catalog (`EXP_ID`)

| EXP_ID | Name | Description |
|---:|---|---|
| 0 | All | run all experiment blocks |
| 1 | Baseline | diffusion baseline sweep |
| 2 | PowerLaw Ablation | sweeps power order / coef init / diag mode / lr / dropout |
| 3 | MixPropDual | DynGCN dual-graph propagation sweep |
| 4 | Chebyshev | Chebyshev sweep |
| 5 | NoDiagonal | remove diagonal (`GWN_DIAG_MODE=neighbor`) |
| 6 | PowerMixDual | DynGCN PowerMixDual sweep (order/init/k/temp/diag) |
| 7 | PowerMixDual Layers Ablation | sweeps `layers` (blocks fixed) |
| 8 | PowerMixDual Graph Variants | random base / no-temporal / single-graph variants |

---

## Sweep grids (environment variables)

The runner is fully controlled via environment variables (defaults shown in `run_experiments_ab.sh`).

### Core training controls
- `DEVICE` (default `cuda:0`)
- `EPOCHS` (default `5`)
- `ADJTYPE` (default `doubletransition`)
- `PRINT_EVERY` (default `50`)
- `RESULTS_CSV` (default `./results.csv`)

### Grid lists
- `DATA_LIST` (fallback to `DATA`, default `data/FRANCE`)
- `BATCH_LIST` (fallback to `BATCH`, default `64`)
- `SEQ_LIST` (default `12`)
- `PRED_LIST` (default `12`)
- `LR_LIST` (default `0.001 0.0001 0.00001`)
- `DROPOUT_LIST` (default `0.3`)
- `NHID_LIST` (default `64`)
- `WD_LIST` (default `0.0001`)
- `BLOCKS_LIST` (default `4`)
- `LAYERS_LIST` (default `2`)

Example: 4 datasets × 3 batch sizes × 3 learning rates:
```bash
DATA_LIST="data/SYNTHETIC_EASY data/SYNTHETIC_MEDIUM data/SYNTHETIC_HARD data/SYNTHETIC_VERY_HARD" BATCH_LIST="32 64 128" LR_LIST="0.001 0.0001 0.00001" EXP_ID=1 bash run_experiments_ab.sh
```

---

## DynGCN / ablation switches (environment variables)

Your `train.py` should read these env vars and configure DynGCN accordingly.

> Note: the env var names are kept for backwards compatibility with your runner script.

### Diagonal/self-loop policy
- `GWN_DIAG_MODE=self_and_neighbor` (default) or `neighbor`

### Adaptive adjacency toggle
- `DISABLE_ADAPTADJ=1` → runner does **not** pass `--addaptadj`

### PowerLaw (EXP 2)
- `GWN_USE_POWER=1`
- `GWN_POWER_ORDER` (e.g., 2, 3, 4)
- `GWN_POWER_INIT` in `{plain, decay, softmax}` (initialization strategy)

### Chebyshev (EXP 4)
- `GWN_USE_CHEBY=1`
- `GWN_CHEBY_K` (e.g., 3)

### MixPropDual (EXP 3)
- `GWN_USE_MIXPROP=1`
- `GWN_MIXPROP_K` (e.g., 3)
- `GWN_ADJ_DROPOUT` (e.g., 0.1)
- `GWN_ADJ_TEMP` (e.g., 1.0)

### PowerMixDual (EXP 6–8)
- `GWN_USE_POWERMIX=1`
- `GWN_POWERMIX_K` (e.g., 2, 3)
- `GWN_POWERMIX_DROPOUT` (adj dropout)
- `GWN_POWERMIX_TEMP` (temperature)
- plus PowerLaw vars: `GWN_POWER_ORDER`, `GWN_POWER_INIT`, and `GWN_DIAG_MODE`

### Graph variants (EXP 8)
- `GWN_RANDOM_BASE_GRAPH=1` → randomize base graph
- `GWN_SECOND_GRAPH_FIXED=1` → second graph fixed (no temporal second graph)
- `GWN_DISABLE_SECOND_GRAPH=1` → single-graph mode

---

## wandb logging

The runner sets:

- `WANDB_PROJECT` (default: `GraphWaveNet` — feel free to rename)
- `WANDB_ENTITY` (default empty)
- `WANDB_MODE` (`online` / `offline` / `disabled`)
- `WANDB_DIR` (default: `./wandb_runs`)

Example:
```bash
WANDB_PROJECT=DynGCN-EXCHANGE WANDB_ENTITY=your_entity WANDB_MODE=online WANDB_DIR=./wandb_runs/EXCHANGE_graph RESULTS_CSV=./results_exchange_graph_variants.csv DATA_LIST="data/EXCHANGE" EXP_ID=8 DEVICE=cuda:0 bash run_experiments_ab.sh
```

---

## Outputs

### CSV results
The runner exports `RESULTS_CSV` (default `./results.csv`).  
Your `train.py` or `_wandb_proxy.py` should append one row per run.

### Named wandb runs
Run names are auto-generated to encode key hyperparameters, e.g.:

- `Baseline_dataEXCHANGE_bs64_seq12_pred12_lr0.001_do0.3_hid64_wd0.0001_b4_l2`
- `PowerLaw_o2_softmax_neighbor_dataEXCHANGE_...`
- `PMD_layersAblation_b4_l1_o3_softmax_K2_self_and_neighbor_...`

---

## SLURM example

```bash
srun -p 4090 --gres=gpu:1 -t 4:00:00 --pty bash -i
conda activate <YOUR_ENV>
DEVICE=cuda:0 EXP_ID=6 bash run_experiments_ab.sh
```

---

## Troubleshooting

- **wandb login errors**
  - Run `wandb login` once, or set `WANDB_MODE=offline` / `WANDB_MODE=disabled`.

- **No `results.csv` written**
  - The runner only exports `RESULTS_CSV`. Ensure your `train.py` or `_wandb_proxy.py` writes to it.

- **Too many runs / too slow**
  - Reduce `DATA_LIST`, `BATCH_LIST`, `LR_LIST`, or comment out unused experiment blocks.
  - Avoid `EXP_ID=0` unless you intentionally want all sweeps.

- **Your `train.py` uses different CLI flags**
  - Edit the command construction in `run_experiments_ab.sh` (`CMD=(python train.py ...)`) to match your script.

---




