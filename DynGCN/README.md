
# SeriesGCN: Capturing High-Order Covariance with Message Passing



---

## Getting Started

### 1. Installation
```bash
conda create -n seriesgcn python=3.9
conda activate seriesgcn
pip install -r requirements.txt
````

### 2. Data Preparation

* **Synthetic datasets**: generated with adjustable correlation ratios.
* **Real-world datasets**: Exchange Rate, Electricity, Traffic, BigElectricity (ENTSO-E).

```bash
python scripts/preprocess_dataset.py --src data/SYNTHETIC_EASY/synthetic.csv
```

### 3. Run Training

```bash
# Exchange rate
python train.py --data data/EXCHANGE --device cuda:0 \
  --batch_size 64 --epochs 10 --seq_length 12 --pred_length 12 \
  --learning_rate 0.001 --dropout 0.3 --nhid 64 \
  --gcn_bool --addaptadj --randomadj --adjtype doubletransition

# France
python train_eval.py --data data/FRANCE --device cuda:0 \
  --batch_size 64 --epochs 5 --seq_length 96 --pred_length 12 \
  --learning_rate 0.0005 --dropout 0 --nhid 64 --weight_decay 0.0001 \
  --print_every 50 --gcn_bool --addaptadj --randomadj \
  --adjtype doubletransition --diag_mode neighbor \
  --use_powermix --powermix_k 2 --powermix_dropout 0 \
  --powermix_temp 1.0 --power_order 2 --power_init decay \
  --wandb --wandb_project powermix-traffic --wandb_mode online \
  --wandb_tags "france,powermix,k2"

# Germany
python train_eval.py --data data/GERMANY --device cuda:0 \
  --batch_size 64 --epochs 5 --seq_length 96 --pred_length 12 \
  --learning_rate 0.0005 --dropout 0 --nhid 64 --weight_decay 0.0001 \
  --print_every 50 --gcn_bool --addaptadj --randomadj \
  --adjtype doubletransition --diag_mode neighbor \
  --use_powermix --powermix_k 2 --powermix_dropout 0 \
  --powermix_temp 1.0 --power_order 2 --power_init decay \
  --wandb --wandb_project powermix-traffic --wandb_mode online \
  --wandb_tags "germany,powermix,k2"

# Solar
python train_eval.py --data data/SOLAR --device cuda:0 \
  --batch_size 1 --epochs 5 --seq_length 96 --pred_length 12 \
  --learning_rate 0.0005 --dropout 0 --nhid 64 --weight_decay 0.0001 \
  --print_every 50 --gcn_bool --addaptadj --randomadj \
  --adjtype doubletransition --diag_mode neighbor \
  --use_powermix --powermix_k 2 --powermix_dropout 0 \
  --powermix_temp 1.0 --power_order 2 --power_init decay \
  --wandb --wandb_project powermix-traffic --wandb_mode online \
  --wandb_tags "solar,powermix,k2"

```

### 4. Ablation Study

```bash
EPOCHS=10 WANDB_MODE=disabled WANDB_PROJECT=SeriesGCN \
RESULTS_CSV=./results_EXCHANGE.csv \
DATA_LIST="data/EXCHANGE" \
BATCH_LIST="64 128 256" \
LR_LIST="0.001 0.0001 0.00001" \
EXP_ID=1 bash scripts/run_experiments_ab.sh
```

---
