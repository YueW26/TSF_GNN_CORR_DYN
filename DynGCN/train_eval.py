# train_eval.py
# -*- coding: utf-8 -*-
# python train_eval.py --data data/FRANCE --device cuda:0 --batch_size 64 --epochs 5 --seq_length 96 --pred_length 12 --learning_rate 0.0005 --dropout 0 --nhid 64 --weight_decay 0.0001 --print_every 50 --gcn_bool --addaptadj --randomadj --adjtype doubletransition --diag_mode neighbor  --use_powermix --powermix_k 2 --powermix_dropout 0 --powermix_temp 1.0 --power_order 2 --power_init decay 
# python train_eval.py --data data/GERMANY --device cuda:0 --batch_size 64 --epochs 5 --seq_length 96 --pred_length 12 --learning_rate 0.0005 --dropout 0 --nhid 64 --weight_decay 0.0001 --print_every 50 --gcn_bool --addaptadj --randomadj --adjtype doubletransition --diag_mode neighbor  --use_powermix --powermix_k 2 --powermix_dropout 0 --powermix_temp 1.0 --power_order 2 --power_init decay 
# --wandb --wandb_project powermix-traffic --wandb_mode online --wandb_tags "france,powermix,k2"
# python train_eval.py --data data/SOLAR --device cuda:0 --batch_size 1 --epochs 5 --seq_length 96 --pred_length 12 --learning_rate 0.0005 --dropout 0 --nhid 64 --weight_decay 0.0001 --print_every 50 --gcn_bool --addaptadj --randomadj --adjtype doubletransition --diag_mode neighbor  --use_powermix --powermix_k 2 --powermix_dropout 0 --powermix_temp 1.0 --power_order 2 --power_init decay 
# COVID19_CA55 / ELECTRICITY / SOLAR ###### --wandb --wandb_project powermix-traffic --wandb_mode online --wandb_tags "france,powermix,k2"


import os
import sys
import time
import csv
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import util
from engine import trainer

# ====== W&B (optional) ======
try:
    import wandb
except ImportError:
    wandb = None
import pdb
from visualize import save_adj, save_adj_binary, to01, binarize01

# ============================
# CLI
# ============================
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str, default='data/METR-LA', help='data path')
parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx.pkl', help='adj data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--gcn_bool', action='store_true', help='whether to add graph convolution layer')
parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')
parser.add_argument('--seq_length', type=int, default=96, help='input sequence length')
parser.add_argument('--pred_length', type=int, default=12, help='prediction length (output sequence length)')###### 
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes') ###### 
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--save', type=str, default='./garage/metr', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')
parser.add_argument('--run_multiple_experiments', action='store_true', help='run experiments with different sequence lengths')

# === Early stopping ===
parser.add_argument('--early_stop_patience', type=int, default=10,
                    help='patience for early stopping on valid loss')
parser.add_argument('--early_stop_min_delta', type=float, default=0.0,
                    help='new_best < best - min_delta')

# === Enhancement toggles / params (kept for compatibility) ===
parser.add_argument('--enhance', type=str, default='none',
                    choices=['none', 'series', 'graph', 'both'],
                    help='enhancement module: series (time), graph (spectral), both, or none')
parser.add_argument('--series_kernel', type=int, default=25,
                    help='kernel size for series decomposition (odd number recommended)')
parser.add_argument('--graph_mode', type=str, default='lowpass',
                    choices=['lowpass', 'highpass', 'none'],
                    help='graph filtering mode on inputs')
parser.add_argument('--graph_alpha', type=float, default=0.5,
                    help='graph filter strength alpha')

# ==== GraphWaveNet / Power / MixProp / Cheby / PowerMix ====
parser.add_argument("--use_power", action="store_true", help="Enable PowerLaw propagation")
parser.add_argument("--use_cheby", action="store_true", help="Enable Chebyshev propagation")
parser.add_argument("--use_mixprop", action="store_true", help="Enable MixPropDual")
parser.add_argument("--use_powermix", action="store_true", help="Enable PowerMixDual")

# Structure related
parser.add_argument("--diag_mode", type=str, default="self_and_neighbor",
                    choices=["self_and_neighbor", "neighbor"], help="diagonal edge mode")

# PowerLaw specific
parser.add_argument("--power_order", type=int, default=2, help="PowerLaw max order")
parser.add_argument("--power_init", type=str, default="plain",
                    choices=["plain", "decay", "softmax"], help="power coef init")

# Chebyshev
parser.add_argument("--cheby_k", type=int, default=3, help="Chebyshev K")

# MixPropDual
parser.add_argument("--mixprop_k", type=int, default=3, help="MixPropDual steps")
parser.add_argument("--adj_dropout", type=float, default=0.1, help="adjacency dropout")
parser.add_argument("--adj_temp", type=float, default=1.0, help="adjacency temperature")

# PowerMixDual
parser.add_argument("--powermix_k", type=int, default=3, help="PowerMix steps")
parser.add_argument("--powermix_dropout", type=float, default=0.3, help="PowerMix A-dropout")
parser.add_argument("--powermix_temp", type=float, default=1.0, help="PowerMix temperature")

# --- Weights & Biases (W&B) ---
parser.add_argument('--wandb', action='store_true', help='enable Weights & Biases logging')
parser.add_argument('--wandb_project', type=str, default='powermix-traffic', help='W&B project')
parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity (team/user); None = default')
parser.add_argument('--wandb_run_name', type=str, default=None, help='Optional custom run name')
parser.add_argument('--wandb_tags', type=str, default='', help='Comma-separated tags, e.g. "france,powermix"')
parser.add_argument('--wandb_mode', type=str, default='online', choices=['online','offline','disabled'],
                    help='online=cloud, offline=local files, disabled=no logging')

args = parser.parse_args()

# ============================
# Dataset auto-config
# ============================
def configure_dataset_params(args):
    data_path = args.data.upper()

    if 'FRANCE' in data_path:
        args.num_nodes = 10
        args.adjdata = 'data/sensor_graph/adj_mx_france.pkl'
        args.save = './garage/france/'
        print("France")
        print(f"n={args.num_nodes}, a阵={args.adjdata}")

    elif 'GERMANY' in data_path:
        args.num_nodes = 16
        args.adjdata = 'data/sensor_graph/adj_mx_germany.pkl'
        args.save = './garage/germany/'
        print("Germany")
        print(f"n={args.num_nodes}, a={args.adjdata}")

    elif 'BAY' in data_path:
        args.num_nodes = 325
        args.adjdata = 'data/sensor_graph/adj_mx_bay.pkl'
        args.save = './garage/bay/'
        print("PEMS-BAY")
        print(f"n={args.num_nodes}, a={args.adjdata}")

    elif 'SYNTHETIC_EASY' in data_path:
        args.num_nodes = 12
        args.adjdata = 'data/sensor_graph/adj_mx_synthetic_easy.pkl'
        args.save = './garage/synth_easy/'
        print("SYNTHETIC_EASY")
        print(f"n={args.num_nodes}, a={args.adjdata}")

    elif 'SYNTHETIC_MEDIUM' in data_path:
        args.num_nodes = 12
        args.adjdata = 'data/sensor_graph/adj_mx_synthetic_medium.pkl'
        args.save = './garage/synth_medium/'
        print("SYNTHETIC_MEDIUM")
        print(f"n={args.num_nodes}, a={args.adjdata}")

    elif 'SYNTHETIC_HARD' in data_path:
        args.num_nodes = 12
        args.adjdata = 'data/sensor_graph/adj_mx_synthetic_hard.pkl'
        args.save = './garage/synth_hard/'
        print("SYNTHETIC_HARD")
        print(f"n={args.num_nodes}, a={args.adjdata}")

    elif 'SYNTHETIC_VERY_HARD' in data_path:
        args.num_nodes = 12
        args.adjdata = 'data/sensor_graph/adj_mx_synthetic_very_hard.pkl'
        args.save = './garage/synth_very_hard/'
        print("SYNTHETIC_VERY_HARD")
        print(f"n={args.num_nodes}, a={args.adjdata}")

    elif 'SOLAR' in data_path:
        args.num_nodes = 137
        args.adjdata = 'data/sensor_graph/adj_mx_solar.pkl'
        args.save = './garage/solar/'
        print("Solar")
        print(f"n={args.num_nodes}, a={args.adjdata}")

    elif 'ELECTRICITY' in data_path:
        args.num_nodes = 321
        args.adjdata = 'data/sensor_graph/adj_mx_electricity.pkl'
        args.save = './garage/electricity/'
        print("Electricity")
        print(f"n={args.num_nodes}, a={args.adjdata}")
        
    elif 'COVID19_CA55' in data_path:
        args.num_nodes = 55
        args.adjdata = 'data/sensor_graph/adj_mx_covid19_CA55.pkl'
        args.save = './garage/covid19/'
        print("Covid19-CA55")
        print(f"n={args.num_nodes}, a={args.adjdata}")


    else:
        print(f"no: {data_path}")
        print(f"n={args.num_nodes}, a={args.adjdata}")

    os.makedirs(args.save, exist_ok=True)
    return args

args = configure_dataset_params(args)

# ============================
# W&B init after args configuration
# ============================
def _get_lr_safe(obj):
    opt = getattr(obj, 'optimizer', None)
    if opt and len(opt.param_groups) > 0 and 'lr' in opt.param_groups[0]:
        return opt.param_groups[0]['lr']
    return None

WANDB_ON = bool(args.wandb and (wandb is not None) and args.wandb_mode != 'disabled')
if WANDB_ON:
    os.environ['WANDB_MODE'] = args.wandb_mode  # 'online' or 'offline'
    run_name = args.wandb_run_name or f"{os.path.basename(args.data)}_seq{args.seq_length}_pred{args.pred_length}_exp{args.expid}"
    tags = [t.strip() for t in args.wandb_tags.split(',') if t.strip()]
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, tags=tags, config=vars(args))

# ============================
# Utilities for multi-run
# ============================
def run_experiments_with_different_seq_lengths():
    seq_lengths = [6, 12]
    results = []

    print("Starting experiments with different sequence lengths...")
    print(f"Sequence lengths to test: {seq_lengths}")
    print("Note: pred_length will be set equal to seq_length for each experiment")

    for seq_len in seq_lengths:
        print("\n" + "="*60)
        print(f"Starting experiment with seq_length = {seq_len}, pred_length = {seq_len}")
        print("="*60)

        args.seq_length = seq_len
        args.pred_length = seq_len
        args.expid = seq_len

        print(f"为 seq_length={seq_len}, pred_length={seq_len} ...")
        generate_data_for_seq_length(seq_len, seq_len)

        experiment_start_time = time.time()
        result = main_experiment()
        experiment_end_time = time.time()

        result['seq_length'] = seq_len
        result['pred_length'] = seq_len
        result['total_experiment_time'] = experiment_end_time - experiment_start_time
        results.append(result)

        print(f"Completed experiment with seq_length = {seq_len}, pred_length = {seq_len}")
        print(f"Experiment time: {experiment_end_time - experiment_start_time:.4f} seconds")

    save_results_to_csv(results)
    print("\nAll experiments completed! Results saved to 'experiment_results.csv'")
    return results

def generate_data_for_seq_length(seq_length, pred_length):
    import subprocess

    data_path = args.data.upper()

    if 'FRANCE' in data_path:
        dataset_name = 'FRANCE'
        process_script = 'process_france_with_dataloader.py'
        data_file = f'data/FRANCE/train.npz'
    elif 'GERMANY' in data_path:
        dataset_name = 'GERMANY'
        process_script = 'process_germany_with_dataloader.py'
        data_file = f'data/GERMANY/train.npz'
    else:
        print(f"new: {data_path}")
        print("use")
        return

    regenerate = True

    if os.path.exists(data_file):
        try:
            data = np.load(data_file)
            existing_seq_len = data['x'].shape[1]
            if existing_seq_len == seq_length:
                print(f"{dataset_name}  (seq_length={seq_length})")
                regenerate = False
            else:
                print(f" {dataset_name}  ({existing_seq_len} != {seq_length})，...")
        except Exception as e:
            print(f" {dataset_name}: {e}")

    if regenerate:
        print(f"{dataset_name}: seq_length={seq_length}, pred_length={pred_length}...")
        cmd = [
            sys.executable, process_script,
            '--step', 'process',
            '--seq_length', str(seq_length),
            '--pred_length', str(pred_length)
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"{dataset_name}")
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[-5:]:
                if line.strip():
                    print(f"  {line}")
        except subprocess.CalledProcessError as e:
            print(f"{dataset_name}: {e}")
            print(f"e: {e.stderr}")
            raise








# ============================
# Main experiment
# ============================
def main_experiment():
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]
    aptinit = None
    if args.aptonly:
        supports = None

    engine = trainer(
        scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
        args.learning_rate, args.weight_decay, args.device, supports, args.gcn_bool,
        args.addaptadj, aptinit, pred_length=args.pred_length,
        diag_mode=args.diag_mode,
        use_power=args.use_power, power_order=args.power_order, power_init=args.power_init,
        use_cheby=args.use_cheby, cheby_k=args.cheby_k,
        use_mixprop=args.use_mixprop, mixprop_k=args.mixprop_k,
        adj_dropout=args.adj_dropout, adj_temp=args.adj_temp,
        use_powermix=args.use_powermix, powermix_k=args.powermix_k,
        powermix_dropout=args.powermix_dropout, powermix_temp=args.powermix_temp
    )

    if WANDB_ON:
        try:
            wandb.watch(engine.model, log='gradients', log_freq=max(1, args.print_every))
        except Exception:
            pass

    print("start training...", flush=True)

    his_loss = []
    val_time = []
    train_time = []

    best_val = float('inf')
    epochs_no_improve = 0
    global_step = 0

    for i in range(1, args.epochs+1):
        # 
        train_loss, train_mae, train_mape, train_rmse, train_rse, train_corr = [], [], [], [], [], []
        valid_loss, valid_mae, valid_mape, valid_rmse, valid_rse, valid_corr = [], [], [], [], [], []

        # === Training ===
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device).transpose(1, 3)  # [B, C, N, T]
            trainy = torch.Tensor(y).to(device).transpose(1, 3)

            mae, mape, rmse, rse, corr, loss = engine.train(
                trainx, trainy[:, 0, :, :args.pred_length]
            )

            train_mae.append(mae)
            train_mape.append(mape)
            train_rmse.append(rmse)
            train_rse.append(rse)
            train_corr.append(corr)
            train_loss.append(loss)

            if WANDB_ON and (iter % args.print_every == 0):
                lr = _get_lr_safe(engine)
                payload = {
                    'iter/train_loss': loss,
                    'iter/train_mae': mae,
                    'iter/train_mape': mape,
                    'iter/train_rmse': rmse,
                    'iter/train_rse': rse,
                    'iter/train_corr': corr,
                    'iter/epoch': i,
                    'iter/step_in_epoch': iter
                }
                if lr is not None:
                    payload['iter/lr'] = lr
                wandb.log(payload, step=global_step)

            global_step += 1

            if iter % args.print_every == 0:
                print(f"Iter: {iter:03d}, Train Loss: {loss:.4f}, "
                      f"Train MAE: {mae:.4f}, Train MAPE: {mape:.4f}, "
                      f"Train RMSE: {rmse:.4f}, Train RSE: {rse:.4f}, Train Corr: {corr:.4f}",
                      flush=True)

        t2 = time.time()
        train_time.append(t2-t1)

        # === Validation ===
        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device).transpose(1, 3)
            testy = torch.Tensor(y).to(device).transpose(1, 3)

            mae, mape, rmse, rse, corr, loss = engine.eval(
                testx, testy[:, 0, :, :args.pred_length]
            )

            valid_mae.append(mae)
            valid_mape.append(mape)
            valid_rmse.append(rmse)
            valid_rse.append(rse)
            valid_corr.append(corr)
            valid_loss.append(loss)

        s2 = time.time()
        print(f"Epoch: {i:03d}, Inference Time: {s2-s1:.4f} secs")
        val_time.append(s2-s1)

        # === Epoch averages ===
        mtrain_mae, mtrain_mape, mtrain_rmse, mtrain_rse, mtrain_corr, mtrain_loss = \
            np.mean(train_mae), np.mean(train_mape), np.mean(train_rmse), np.mean(train_rse), np.mean(train_corr), np.mean(train_loss)
        mvalid_mae, mvalid_mape, mvalid_rmse, mvalid_rse, mvalid_corr, mvalid_loss = \
            np.mean(valid_mae), np.mean(valid_mape), np.mean(valid_rmse), np.mean(valid_rse), np.mean(valid_corr), np.mean(valid_loss)

        his_loss.append(mvalid_loss)

        print(('Epoch: {:03d}, '
               'Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Train RSE: {:.4f}, Train Corr: {:.4f}, Train Loss: {:.4f}, '
               'Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Valid RSE: {:.4f}, Valid Corr: {:.4f}, Valid Loss: {:.4f}, '
               'Training Time: {:.4f}/epoch')
              .format(i,
                      mtrain_mae, mtrain_mape, mtrain_rmse, mtrain_rse, mtrain_corr, mtrain_loss,
                      mvalid_mae, mvalid_mape, mvalid_rmse, mvalid_rse, mvalid_corr, mvalid_loss,
                      (t2 - t1)),
              flush=True)

        # adjacency 
        torch.save(engine.model.powermix_convs[0].adj_1, f"adj1_{i}.pt") 
        torch.save(engine.model.powermix_convs[0].adj_2, f"adj2_{i}.pt") 

        # W&B per-epoch logging
        if WANDB_ON:
            lr = _get_lr_safe(engine)
            payload = {
                'epoch': i,
                'train/loss': mtrain_loss,
                'train/mae': mtrain_mae,
                'train/mape': mtrain_mape,
                'train/rmse': mtrain_rmse,
                'train/rse': mtrain_rse,
                'train/corr': mtrain_corr,
                'valid/loss': mvalid_loss,
                'valid/mae': mvalid_mae,
                'valid/mape': mvalid_mape,
                'valid/rmse': mvalid_rmse,
                'valid/rse': mvalid_rse,
                'valid/corr': mvalid_corr,
                'time/train_epoch_sec': (t2 - t1),
                'time/valid_epoch_sec': (s2 - s1)
            }
            if lr is not None:
                payload['lr'] = lr
            wandb.log(payload, step=global_step)

        # === Early stopping ===
        if mvalid_loss < best_val - args.early_stop_min_delta:
            best_val = mvalid_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= args.early_stop_patience:
            print(f"Early stopping at epoch {i}. "
                  f"Best valid loss: {best_val:.4f} (epoch {np.argmin(his_loss) + 1}).")
            break

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # === Testing ===
    bestid = int(np.argmin(his_loss))
    best_ckpt = args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid], 2))+".pth"
    engine.model.load_state_dict(torch.load(best_ckpt))

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :args.pred_length]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device).transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1, 3)
        if len(preds.shape) == 4:
            preds = preds[:, 0, :, :]
        outputs.append(preds)
        del testx, preds
        torch.cuda.empty_cache()

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))

    amae, amape, armse, arse, acorr, horizon_results = [], [], [], [], [], []
    for i in range(args.pred_length):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        mae, mape, rmse, rse, corr = util.metric(pred, real)

        print(f'Evaluate best model on test data for horizon {i+1:02d}, '
              f'Test MAE: {mae:.4f}, Test MAPE: {mape:.4f}, Test RMSE: {rmse:.4f}, '
              f'Test RSE: {rse:.4f}, Test Corr: {corr:.4f}')

        amae.append(mae)
        amape.append(mape)
        armse.append(rmse)
        arse.append(rse)
        acorr.append(corr)
        horizon_results.append({'horizon': i+1, 'mae': mae, 'mape': mape, 'rmse': rmse, 'rse': rse, 'corr': corr})

    avg_mae, avg_mape, avg_rmse, avg_rse, avg_corr = \
        float(np.mean(amae)), float(np.mean(amape)), float(np.mean(armse)), float(np.mean(arse)), float(np.mean(acorr))

    print(f'On average over {args.pred_length:d} horizons, '
          f'Test MAE: {avg_mae:.4f}, Test MAPE: {avg_mape:.4f}, Test RMSE: {avg_rmse:.4f}, '
          f'Test RSE: {avg_rse:.4f}, Test Corr: {avg_corr:.4f}')

    if WANDB_ON:
        wandb.summary['best/valid_loss'] = float(his_loss[bestid])
        wandb.summary['test/mae_avg'] = avg_mae
        wandb.summary['test/mape_avg'] = avg_mape
        wandb.summary['test/rmse_avg'] = avg_rmse
        wandb.summary['test/rse_avg'] = avg_rse
        wandb.summary['test/corr_avg'] = avg_corr
        wandb.summary['train/time_sec_per_epoch_avg'] = float(np.mean(train_time))
        wandb.summary['valid/time_sec_avg'] = float(np.mean(val_time))
        wandb.save(best_ckpt)

    torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")

    return {
        'valid_loss_best': his_loss[bestid],
        'avg_train_time_per_epoch': np.mean(train_time),
        'avg_inference_time': np.mean(val_time),
        'test_mae_avg': avg_mae,
        'test_mape_avg': avg_mape,
        'test_rmse_avg': avg_rmse,
        'test_rse_avg': avg_rse,
        'test_corr_avg': avg_corr,
        'horizon_results': horizon_results
    }



# ============================
# Save CSVs for multi-run
# ============================
def save_results_to_csv(results):
    csv_data = []
    for result in results:
        row = {
            'seq_length': result['seq_length'],
            'pred_length': result['pred_length'],
            'total_experiment_time': result['total_experiment_time'],
            'valid_loss_best': result['valid_loss_best'],
            'avg_train_time_per_epoch': result['avg_train_time_per_epoch'],
            'avg_inference_time': result['avg_inference_time'],
            'test_mae_avg': result['test_mae_avg'],
            'test_mape_avg': result['test_mape_avg'],
            'test_rmse_avg': result['test_rmse_avg'],
            'test_rse_avg': result['test_rse_avg'],
            'test_corr_avg': result['test_corr_avg']
        }
        for hr in result['horizon_results']:
            row[f'horizon_{hr["horizon"]}_mae']  = hr['mae']
            row[f'horizon_{hr["horizon"]}_mape'] = hr['mape']
            row[f'horizon_{hr["horizon"]}_rmse'] = hr['rmse']
            row[f'horizon_{hr["horizon"]}_rse']  = hr['rse']
            row[f'horizon_{hr["horizon"]}_corr'] = hr['corr']
        csv_data.append(row)

    df = pd.DataFrame(csv_data)
    df.to_csv('experiment_results.csv', index=False)
    print("\nResults saved to 'experiment_results.csv'")
    print(f"Columns saved: {list(df.columns)}")

    summary_data = []
    for result in results:
        summary_data.append({
            'seq_length': result['seq_length'],
            'pred_length': result['pred_length'],
            'total_time_hours': result['total_experiment_time'] / 3600,
            'valid_loss': result['valid_loss_best'],
            'test_mae': result['test_mae_avg'],
            'test_mape': result['test_mape_avg'],
            'test_rmse': result['test_rmse_avg'],
            'test_rse': result['test_rse_avg'],
            'test_corr': result['test_corr_avg']
        })
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('experiment_summary.csv', index=False)
    print("Summary saved to 'experiment_summary.csv'")
    
    
# ============================
# Entrypoint
# ============================
if __name__ == "__main__":
    total_start_time = time.time()

    if args.run_multiple_experiments:
        print("multi...")
        results = run_experiments_with_different_seq_lengths()
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        print("\nAll experiments completed!")
        print(f"Total time for all experiments: {total_time:.4f} seconds ({total_time/3600:.2f} hours)")
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        for result in results:
            print(f"Seq Length {result['seq_length']:2d}, Pred Length {result['pred_length']:2d}: "
                  f"MAE={result['test_mae_avg']:.4f}, "
                  f"MAPE={result['test_mape_avg']:.4f}, "
                  f"RMSE={result['test_rmse_avg']:.4f}, "
                  f"RSE={result['test_rse_avg']:.4f}, "
                  f"Corr={result['test_corr_avg']:.4f}")
    else:
        print("single...")
        print(f"a: seq_length={args.seq_length}, pred_length={args.pred_length}, enhance={args.enhance}")
        print(f"for seq_length={args.seq_length}, pred_length={args.pred_length} ...")
        result = main_experiment()
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        print("\nExperiment completed!")
        print(f"Total time: {total_time:.4f} seconds ({total_time/3600:.2f} hours)")
        print(f"Results: "
              f"MAE={result['test_mae_avg']:.4f}, "
              f"MAPE={result['test_mape_avg']:.4f}, "
              f"RMSE={result['test_rmse_avg']:.4f}, "
              f"RSE={result['test_rse_avg']:.4f}, "
              f"Corr={result['test_corr_avg']:.4f}")

    if WANDB_ON:
        wandb.finish()

