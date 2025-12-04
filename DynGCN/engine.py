
import torch.optim as optim
from model2 import gwnet  ######
import util
import torch
import torch.nn as nn



class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid, dropout, lrate, wdecay,
                device, supports, gcn_bool, addaptadj, aptinit, pred_length=12,
                #
                diag_mode="self_and_neighbor",
                use_power=False, power_order=2, power_init="plain",
                use_cheby=False, cheby_k=3,
                use_mixprop=False, mixprop_k=3, adj_dropout=0.1, adj_temp=1.0,
                use_powermix=False, powermix_k=3, powermix_dropout=0.3, powermix_temp=1.0):
        self.model = gwnet(
            device=device,
            num_nodes=num_nodes,
            dropout=dropout,
            supports=supports,
            gcn_bool=gcn_bool,
            addaptadj=addaptadj,
            aptinit=aptinit,
            in_dim=in_dim,
            out_dim=pred_length,
            residual_channels=nhid,
            dilation_channels=nhid,
            skip_channels=nhid * 8,
            end_channels=nhid * 16,
            # 
            diag_mode=diag_mode,
            use_power=use_power,
            power_order=power_order,
            power_init=power_init,
            use_cheby=use_cheby,
            cheby_k=cheby_k,
            use_mixprop=use_mixprop,
            mixprop_k=mixprop_k,
            adj_dropout=adj_dropout,
            adj_temp=adj_temp,
            use_powermix=use_powermix,
            powermix_k=powermix_k,
            powermix_dropout=powermix_dropout,
            powermix_temp=powermix_temp
        )
        
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()

        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)
        output = output.transpose(1, 3)
        # output: [batch_size, 12, num_nodes, 1]

        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)

        # optimize
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        # metrics (no grad)
        
        # with torch.no_grad():
        #     mape = util.masked_mape(predict, real, 0.0).item()
        #     mae  = util.masked_mae(predict,  real, 0.0).item()
        #     rmse = util.masked_rmse(predict, real, 0.0).item()
        #     rse  = util.masked_rse(predict,  real).item()
        #     corr = util.masked_corr(predict, real).item()
        # return loss.item(), mape, mae, rmse, rse, corr
        
        with torch.no_grad():
            mae  = util.masked_mae(predict,  real, 0.0).item()
            mape = util.masked_mape(predict, real, 0.0).item()
            rmse = util.masked_rmse(predict, real, 0.0).item()
            rse  = util.masked_rse(predict,  real).item()
            corr = util.masked_corr(predict, real).item()
        return mae, mape, rmse, rse, corr, loss.item()


    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input)
        output = output.transpose(1, 3)
        # output = [batch_size, 12, num_nodes, 1]
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)

        # basic loss
        loss = self.loss(predict, real, 0.0)

        # metrics
        
        # mape = util.masked_mape(predict, real, 0.0).item()
        # mae = util.masked_mae(predict, real, 0.0).item()
        # rmse = util.masked_rmse(predict, real, 0.0).item()
        # rse = util.masked_rse(predict, real).item()
        # corr = util.masked_corr(predict, real).item()
        # return loss.item(), mape, mae, rmse, rse, corr
    
        mae  = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        rse  = util.masked_rse(predict, real).item()
        corr = util.masked_corr(predict, real).item()
        return mae, mape, rmse, rse, corr, loss.item()

