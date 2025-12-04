import os
import torch
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'MambaSimple': MambaSimple,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

'''
import numpy as np
import torch
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader
from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

import argparse

# 定义超参数搜索的空间
param_grid = {
    'learning_rate': [1e-5, 1e-4, 1e-3],
    'batch_size': [8, 16, 32],
    'e_layers': [2, 3],
    'd_layers': [1, 2],
    'pred_len': [24, 48, 96],
    'd_model': [256, 512],
}

# 定义搜索方法 (网格搜索)
def objective(params, args):
    # 设置参数
    args.learning_rate = params['learning_rate']
    args.batch_size = params['batch_size']
    args.e_layers = params['e_layers']
    args.d_layers = params['d_layers']
    args.pred_len = params['pred_len']
    args.d_model = params['d_model']
    
    # 创建实验对象
    exp = Exp_Long_Term_Forecast(args)
    
    # 训练模型
    exp.train(f"tuning_lr{params['learning_rate']}_bs{params['batch_size']}")
    
    # 测试模型并获取结果 (可以根据实际情况调整评估方式)
    test_loss = exp.test(f"tuning_lr{params['learning_rate']}_bs{params['batch_size']}")
    
    return test_loss

# 定义训练的主函数
def run_hyperparameter_search(args):
    best_params = None
    best_loss = float('inf')
    
    # 使用网格搜索遍历所有超参数组合
    param_combinations = list(ParameterGrid(param_grid))

    for params in param_combinations:
        # 计算当前超参数组合的损失
        print(f"Evaluating params: {params}")
        current_loss = objective(params, args)
        
        # 更新最佳参数
        if current_loss < best_loss:
            best_loss = current_loss
            best_params = params
            print(f"New best loss: {best_loss}, params: {best_params}")
    
    print(f"Best hyperparameters: {best_params}")
    return best_params


import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Time-Series Forecasting')
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--root_path', type=str, default='/Users/wangbo/Documents/Time-Series-Library-git/Time-Series-Library-1/datasets')
    parser.add_argument('--data_path', type=str, default='Merged_cleaned_germany.csv')
    parser.add_argument('--model_id', type=str, default='Merged_Data_Transformer')
    parser.add_argument('--model', type=str, default='DLinear')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--factor', type=int, default=3)
    parser.add_argument('--enc_in', type=int, default=18)
    parser.add_argument('--dec_in', type=int, default=18)
    parser.add_argument('--c_out', type=int, default=18)
    parser.add_argument('--des', type=str, default='Merged_Data_Transformer')
    parser.add_argument('--seq_len', type=int, default=48)
    parser.add_argument('--label_len', type=int, default=24)
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--train_epochs', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--target', type=str, default='Day-ahead Price [EUR/MWh]')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')  # 禁用GPU使用

    # 直接从launch.json传入的命令行参数中解析参数
    args = parser.parse_args()

    print("Parsed arguments:", args)

    # 运行超参数搜索或其他任务
    best_hyperparams = run_hyperparameter_search(args)
'''