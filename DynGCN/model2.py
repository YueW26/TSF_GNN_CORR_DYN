import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from visualize import save_adj, save_adj_binary, to01, binarize01


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        # x: [B, C, N, T], A: [N, N]
        return torch.einsum("ncvl,vw->ncwl", (x, A)).contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = nn.Conv2d(
            c_in, c_out, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True
        )

    def forward(self, x):
        return self.mlp(x)


class ChebConv(nn.Module):

    def __init__(self, c_in, c_out, K=3, dropout=0.0):
        super(ChebConv, self).__init__()
        assert K >= 1
        self.K = K
        self.dropout = dropout
        self.theta = nn.ModuleList(
            [nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=True) for _ in range(K)]
        )
        self.alpha = nn.Parameter(torch.ones(K), requires_grad=True)
        self.last_cheb_alphas = None  # 

    @staticmethod
    def build_laplacian(A, add_self=True, eps=1e-5):
        N = A.size(0)
        if add_self:
            A = A + torch.eye(N, device=A.device)
        deg = torch.clamp(A.sum(-1), min=eps)
        D_inv_sqrt = torch.diag(torch.pow(deg, -0.5))
        L = torch.eye(N, device=A.device) - D_inv_sqrt @ A @ D_inv_sqrt
        return L

    def forward(self, x, A, include_self=True):
        B, C, N, T = x.shape
        L = self.build_laplacian(A, add_self=include_self)
        L_tilde = 2.0 * L - torch.eye(N, device=A.device)

        Tx_list = []
        Tx0 = x
        Tx_list.append(Tx0)
        if self.K > 1:
            Tx1 = torch.einsum("vw,bcwl->bcvl", L_tilde, x)
            Tx_list.append(Tx1)
            for _k in range(2, self.K):
                Tx2 = 2 * torch.einsum("vw,bcwl->bcvl", L_tilde, Tx1) - Tx0
                Tx_list.append(Tx2)
                Tx0, Tx1 = Tx1, Tx2

        out = 0
        with torch.no_grad():
            self.last_cheb_alphas = self.alpha.detach().cpu()

        for k in range(self.K):
            out = out + self.alpha[k] * self.theta[k](Tx_list[k])

        out = F.dropout(out, self.dropout, training=self.training)
        return out


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2,
                 use_power=False, diag_mode="self_and_neighbor"):
        super(gcn, self).__init__()
        self.nconv = nconv()
        self.dropout = dropout
        self.order = order
        self.support_len = support_len
        self.use_power = use_power
        assert diag_mode in ["neighbor", "self_and_neighbor"]
        self.diag_mode = diag_mode

        c_total = (order * support_len + 1) * c_in
        self.mlp = linear(c_total, c_out)

        # 
        self.power_coef = nn.Parameter(torch.ones(order), requires_grad=True)
        self.last_power_coef = None

    def _apply_diag_policy(self, A):
        if self.diag_mode == "neighbor":
            A = A.clone()
            A.fill_diagonal_(0.0)
            return A
        else:
            return A

    def _matrix_powers(self, A, K):
        powers = []
        Ak = A
        for _ in range(K):
            powers.append(Ak)
            Ak = Ak @ A
        return powers

    def forward(self, x, supports):
        # with torch.no_grad():
        #     self.last_power_coef = self.power_coef.detach().cpu()

        out = [x]
        for A in supports:
            A_use = self._apply_diag_policy(A)
            if self.use_power: # power law
                # A^k
                A_pows = self._matrix_powers(A_use, self.order)
                for k_idx, Ak in enumerate(A_pows):
                    xk = self.nconv(x, Ak)
                    out.append(self.power_coef[k_idx] * xk)
                    
            else:
                # diffusion
                x1 = self.nconv(x, A_use)
                out.append(x1)
                x_prev = x1
                for _k in range(2, self.order + 1):
                    x2 = self.nconv(x_prev, A_use)
                    out.append(x2)
                    x_prev = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


# =========================
# MixPropDual
# =========================

class MixPropDual(nn.Module):

    def __init__(self, c_in, c_out, K=3, droprate=0.1, temperature=1.0,
                 diag_mode='self_and_neighbor', 
                 r_dim=10, 
                 emb_dim=16,
                 use_laplacian=False,
                 embed_init='xavier'):
        super().__init__()
        assert K >= 1
        self.K = K
        self.droprate = float(droprate)
        self.temperature = float(temperature)
        assert diag_mode in ['neighbor', 'self_and_neighbor']
        self.diag_mode = diag_mode
        self.use_laplacian = use_laplacian

        self.k_convs = nn.ModuleList([nn.Conv2d(c_in, c_out, kernel_size=1) for _ in range(K)])
        self.gates = nn.Parameter(torch.ones(K), requires_grad=True)

        # self.r1 = nn.Parameter(torch.randn(1, r_dim), requires_grad=True)
        ###--------------------------------------------------------------------------------------###
        # self.r1 = nn.Embedding(r_dim, emb_dim) ###
        # if embed_init == 'normal': ###
        #     nn.init.normal_(self.r1.weight, mean=0.0, std=0.02) ###
        # else: ###
        #     # xavier ###
        #     nn.init.xavier_uniform_(self.r1.weight) ###
        ###--------------------------------------------------------------------------------------###
        self.src_emb = nn.Embedding(r_dim, emb_dim)
        self.dst_emb = nn.Embedding(r_dim, emb_dim)

        if embed_init == 'normal':
            nn.init.normal_(self.src_emb.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.dst_emb.weight, mean=0.0, std=0.02)
        else:
            nn.init.xavier_uniform_(self.src_emb.weight)
            nn.init.xavier_uniform_(self.dst_emb.weight)
        ###--------------------------------------------------------------------------------------###
        self.adj_1 = None
        self.adj_2 = None
        
    @staticmethod
    def _apply_diag(A, mode):
        if mode == 'neighbor':
            A = A.clone()
            A.fill_diagonal_(0.0)
            return A
        return A

    @staticmethod
    def _laplacian(A, eps=1e-6):
        N = A.size(0)
        deg = torch.clamp(A.sum(-1), min=eps)
        D_inv_sqrt = torch.diag(torch.pow(deg, -0.5))
        L = torch.eye(N, device=A.device) - D_inv_sqrt @ A @ D_inv_sqrt
        return L

    def _build_adj1_from_A(self, A):
        M = self._laplacian(A) if self.use_laplacian else A
        M = F.relu(M)
        adj_1 = torch.softmax(M / max(self.temperature, 1e-6), dim=1)
        adj_1 = self._apply_diag(adj_1, self.diag_mode)
        if self.training and self.droprate > 0:
            adj_1 = F.dropout(adj_1, p=self.droprate)
        return adj_1

    # def _build_adj2_from_r1(self, N, device):
    #     r1 = self.r1
    #     if r1.size(0) != N:
    #         r1 = r1.expand(N, -1).contiguous()
    #     S = r1 @ r1.t()
    #     S = F.relu(S)
    #     adj_2 = torch.softmax(S / max(self.temperature, 1e-6), dim=1)
    #     adj_2 = self._apply_diag(adj_2, self.diag_mode)
    #     if self.training and self.droprate > 0:
    #         adj_2 = F.dropout(adj_2, p=self.droprate)
    #     return adj_2

    def _build_adj2_from_r1(self, N, device, node_idx: torch.Tensor = None):
     
        if node_idx is None:
            idx = torch.arange(N, device=device, dtype=torch.long)
        else:
            idx = node_idx.to(device=device, dtype=torch.long)

        #  R: [N, r_dim]
        ###--------------------------------------------------------------------------------------###
        # R = self.r1(idx)
        
        # # 
        # if self.use_cosine:
        #     R = F.normalize(R, p=2, dim=1)

        #  S: [N, N]
        # S = R @ R.t()
        ###--------------------------------------------------------------------------------------###
        U = self.src_emb(idx)   # [N, emb_dim]
        V = self.dst_emb(idx)   # [N, emb_dim]

        #  S: [N, N]
        S = U @ V.t()
        ###--------------------------------------------------------------------------------------###
        S = F.relu(S)  #
        # 
        adj_2 = torch.softmax(S / max(self.temperature, 1e-6), dim=1)
        adj_2 = self._apply_diag(adj_2, self.diag_mode)
        if self.training and self.droprate > 0:
            adj_2 = F.dropout(adj_2, p=self.droprate)
        return adj_2

    def forward(self, x, A_base):
        B, C, N, T = x.shape
        device = x.device
        ###--------------------------------------------------------------------------------------###
        # self.adj_1 = self._build_adj1_from_A(A_base)
        # self.adj_2 = self._build_adj2_from_r1(N, device)
        ###--------------------------------------------------------------------------------------###
        self.adj_1 = self._build_adj1_from_A(A_base).detach()
        self.adj_2 = self._build_adj2_from_r1(N, device).detach()
        ###--------------------------------------------------------------------------------------###
        inj = [self.k_convs[k](x) * self.gates[k] for k in range(self.K)]

        # 1
        z = inj[0].permute(0, 3, 2, 1).contiguous().view(B * T, N, -1)
        for k in range(1, self.K):
            z = self.adj_1 @ z + inj[k].permute(0, 3, 2, 1).contiguous().view(B * T, N, -1)
        z = z.view(B, T, N, -1).permute(0, 3, 2, 1).contiguous()

        # 2
        z_fix = inj[0].permute(0, 3, 2, 1).contiguous().view(B * T, N, -1)
        for k in range(1, self.K):
            z_fix = self.adj_2 @ z_fix + inj[k].permute(0, 3, 2, 1).contiguous().view(B * T, N, -1)
        z_fix = z_fix.view(B, T, N, -1).permute(0, 3, 2, 1).contiguous()

        return z + z_fix


# =========================
# PowerMixDual (NEW)
# =========================

class PowerMixDual(nn.Module):

    def __init__(self, c_in, c_out, K=3, droprate=0.1, temperature=1.0,
                 diag_mode='self_and_neighbor', r_dim=10, embed_init='xavier', emb_dim=16):
        super().__init__()
        assert K >= 1
        self.K = K
        self.droprate = float(droprate)
        self.temperature = float(temperature)
        self.diag_mode = diag_mode

        self.k_convs = nn.ModuleList([nn.Conv2d(c_in, c_out, kernel_size=1) for _ in range(K)])
        self.gates = nn.Parameter(torch.ones(K), requires_grad=True)
        self.power_coef = nn.Parameter(torch.ones(K), requires_grad=True)
        self.last_power_coef = None

        ###--------------------------------------------------------------------------------------###
        # self.r1 = nn.Embedding(r_dim, emb_dim)
        # if embed_init == 'normal':
        #     nn.init.normal_(self.r1.weight, mean=0.0, std=0.02)
        # else:
        #     # xavier
        #     nn.init.xavier_uniform_(self.r1.weight)
        ###--------------------------------------------------------------------------------------###
        self.src_emb = nn.Embedding(r_dim, emb_dim)
        self.dst_emb = nn.Embedding(r_dim, emb_dim)

        if embed_init == 'normal':
            nn.init.normal_(self.src_emb.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.dst_emb.weight, mean=0.0, std=0.02)
        else:
            nn.init.xavier_uniform_(self.src_emb.weight)
            nn.init.xavier_uniform_(self.dst_emb.weight)
        ###--------------------------------------------------------------------------------------###
        self.adj_1 = None
        self.adj_2 = None

        
    @staticmethod
    def _apply_diag(A, mode):
        if mode == 'neighbor':
            A = A.clone()
            A.fill_diagonal_(0.0)
            return A
        return A

    def _build_adj_softmax(self, M):
        M = F.relu(M)
        adj = torch.softmax(M / max(self.temperature, 1e-6), dim=1)
        adj = self._apply_diag(adj, self.diag_mode)
        if self.training and self.droprate > 0:
            adj = F.dropout(adj, p=self.droprate)
        return adj

    def _build_adj1_from_A(self, A):
        return self._build_adj_softmax(A)

    def _build_adj2_from_r1(self, N, device):
        ###--------------------------------------------------------------------------------------###
        U = self.src_emb(torch.arange(N, device=device))
        V = self.dst_emb(torch.arange(N, device=device))
        S = U @ V.t()
        ###--------------------------------------------------------------------------------------###
        return self._build_adj_softmax(S)

    def forward(self, x, A_base):
        B, C, N, T = x.shape
        device = x.device
        # with torch.no_grad():
        #     self.last_power_coef = self.power_coef.detach().cpu()

        ###--------------------------------------------------------------------------------------###
        # self.adj_1 = self._build_adj1_from_A(A_base)
        # self.adj_2 = self._build_adj2_from_r1(N, device)
        ###--------------------------------------------------------------------------------------###
        self.adj_1 = self._build_adj1_from_A(A_base).detach()
        self.adj_2 = self._build_adj2_from_r1(N, device).detach()
        ###--------------------------------------------------------------------------------------###
        inj = [self.k_convs[k](x) * self.gates[k] for k in range(self.K)]

        #
        z = inj[0].permute(0, 3, 2, 1).contiguous().view(B * T, N, -1)
        z2 = z.clone()
        for k in range(1, self.K):
            coef = self.power_coef[k]
            z = self.adj_1 @ z + coef * inj[k].permute(0, 3, 2, 1).contiguous().view(B * T, N, -1)
            z2 = self.adj_2 @ z2 + coef * inj[k].permute(0, 3, 2, 1).contiguous().view(B * T, N, -1)

        z = z.view(B, T, N, -1).permute(0, 3, 2, 1).contiguous()
        z2 = z2.view(B, T, N, -1).permute(0, 3, 2, 1).contiguous()

        return z + z2




class gwnet(nn.Module):
    def __init__(self,
                 device,
                 num_nodes,
                 dropout=0.3,
                 supports=None,
                 gcn_bool=True,
                 addaptadj=True,
                 aptinit=None,
                 in_dim=2,
                 out_dim=12,
                 residual_channels=32,
                 dilation_channels=32,
                 skip_channels=256,
                 end_channels=512,
                 kernel_size=2,
                 blocks=4,
                 layers=2,
                 # ==== =========
                 diag_mode: str = "self_and_neighbor",                       # {"self_and_neighbor","neighbor"}
                 # PowerLaw
                 use_power: bool = False,
                 power_order: int = 2,
                 power_init: str = "plain",                                   # {"plain","decay","softmax"}
                 # Chebyshev
                 use_cheby: bool = False,
                 cheby_k: int = 3,
                 # MixPropDual
                 use_mixprop: bool = False,
                 mixprop_k: int = 3,
                 adj_dropout: float = 0.1,
                 adj_temp: float = 1.0,
                 # PowerMixDual
                 use_powermix: bool = False,
                 powermix_k: int = 3,
                 powermix_dropout: float = 0.3,
                 powermix_temp: float = 1.0):
        super(gwnet, self).__init__()
        
        # ====== ====
        self.device = device
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.num_nodes = num_nodes
        self.supports = supports
        self.aptinit = aptinit
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.end_channels = end_channels
        self.kernel_size = kernel_size

        # ====== 
        # diag
        if diag_mode not in {"self_and_neighbor", "neighbor"}:
            raise ValueError(f"diag_mode must be 'self_and_neighbor' or 'neighbor', got {diag_mode}")
        self.diag_mode = diag_mode

        # PowerLaw
        if power_init not in {"plain", "decay", "softmax"}:
            raise ValueError(f"power_init must be one of {{'plain','decay','softmax'}}, got {power_init}")
        self.use_power = bool(use_power)
        self.power_order = int(power_order)
        self.power_init = power_init

        # Chebyshev
        self.use_cheby = bool(use_cheby)
        self.cheby_K = int(cheby_k)

        # MixPropDual
        self.use_mixprop = bool(use_mixprop)
        self.mixprop_K = int(mixprop_k)
        self.adj_droprate = float(adj_dropout)
        self.adj_temperature = float(adj_temp)

        # PowerMixDual
        self.use_powermix = bool(use_powermix)

        self.powermix_K = int(powermix_k)
        self.powermix_droprate = float(powermix_dropout)
        self.powermix_temperature = float(powermix_temp)

        # 
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.cheb_convs = nn.ModuleList()
        self.mixprop_convs = nn.ModuleList()
        self.powermix_convs = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        # 
        receptive_field = 1
        new_dilation = 1
        for _b in range(blocks):
            for _i in range(layers):
                receptive_field += (kernel_size - 1) * new_dilation
                new_dilation *= 2
        self.receptive_field = receptive_field

        # supports
        self.supports = supports
        self.supports_len = len(supports) if supports is not None else 0

        # 
        if gcn_bool and addaptadj:
            if aptinit is None:
                if self.supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True)
                self.supports_len += 1
            else:
                U, S, V = torch.svd(aptinit)
                initemb1 = U[:, :10] @ torch.diag(S[:10].pow(0.5))
                initemb2 = torch.diag(S[:10].pow(0.5)) @ V[:, :10].t()
                self.nodevec1 = nn.Parameter(initemb1.to(device), requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2.to(device), requires_grad=True)
                self.supports_len += 1

        
        # 
        new_dilation = 1
        for _b in range(blocks):
            for _i in range(layers):
                # filter & gate
                self.filter_convs.append(
                    nn.Conv2d(residual_channels, dilation_channels,
                              kernel_size=(1, kernel_size), dilation=new_dilation)
                )
                self.gate_convs.append(
                    nn.Conv2d(residual_channels, dilation_channels,
                              kernel_size=(1, kernel_size), dilation=new_dilation)
                )

                # residual & skip
                self.residual_convs.append(nn.Conv2d(dilation_channels, residual_channels, kernel_size=(1, 1)))
                self.skip_convs.append(nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))

                # spatial
                if gcn_bool:
                    if self.use_cheby:
                        self.cheb_convs.append(
                            ChebConv(dilation_channels, residual_channels,
                                     K=self.cheby_K, dropout=dropout)
                        )
                        self.gconv.append(None)
                        self.mixprop_convs.append(None)
                        self.powermix_convs.append(None)
                    elif self.use_mixprop:
                        self.cheb_convs.append(None)
                        self.gconv.append(None)
                        self.mixprop_convs.append(
                            MixPropDual(
                                c_in=dilation_channels,
                                c_out=residual_channels,
                                K=self.mixprop_K,
                                droprate=self.adj_droprate,
                                temperature=self.adj_temperature,
                                diag_mode=self.diag_mode,
                                r_dim=10,
                                use_laplacian=False
                            )
                        )
                        self.powermix_convs.append(None)
                    elif self.use_powermix:
                        self.cheb_convs.append(None)
                        self.gconv.append(None)
                        self.mixprop_convs.append(None)
                        
                        self.powermix_convs.append(
                            PowerMixDual(
                                c_in=dilation_channels,
                                c_out=residual_channels,
                                K=self.powermix_K,
                                droprate=self.powermix_droprate,
                                temperature=self.powermix_temperature,
                                diag_mode=self.diag_mode,
                                r_dim=10
                            )
                        )
                    else:
                        self.gconv.append(
                            gcn(dilation_channels, residual_channels, dropout,
                                support_len=self.supports_len, order=2,
                                use_power=self.use_power, diag_mode=self.diag_mode)
                        )
                        self.cheb_convs.append(None)
                        self.mixprop_convs.append(None)
                        self.powermix_convs.append(None)
                else:
                    self.gconv.append(None)
                    self.cheb_convs.append(None)
                    self.mixprop_convs.append(None)
                    self.powermix_convs.append(None)

                new_dilation *= 2

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim, kernel_size=(1, 1), bias=True)

    def _build_adaptive_adj(self):
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        return adp

    def _collect_supports(self):
        sup = []
        if self.supports is not None:
            sup += self.supports
        if self.gcn_bool and self.addaptadj:
            sup.append(self._build_adaptive_adj())
        return sup if len(sup) > 0 else None

    def forward(self, input):
        
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = F.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input

        x = self.start_conv(x)
        skip = 0

        new_supports = self._collect_supports() if self.gcn_bool else None

        layer_idx = 0
        for _b in range(self.blocks):
            for _i in range(self.layers):
                residual = x
                filt = torch.tanh(self.filter_convs[layer_idx](residual))
                gate = torch.sigmoid(self.gate_convs[layer_idx](residual))
                x = filt * gate

                # skip
                s = self.skip_convs[layer_idx](x)
                try:
                    skip = skip[:, :, :, -s.size(3):]
                except Exception:
                    skip = 0
                skip = s + skip

                # 
                if self.gcn_bool and (new_supports is not None):
                    if self.cheb_convs[layer_idx] is not None:
                        acc = 0
                        for A in new_supports:
                            xs = self.cheb_convs[layer_idx](x, A, include_self=(self.diag_mode == "self_and_neighbor"))
                            acc = acc + xs
                        x = acc / float(len(new_supports))
                    elif self.mixprop_convs[layer_idx] is not None:
                        if len(new_supports) > 0:
                            A_base = new_supports[0]
                        else:
                            A_base = self._build_adaptive_adj()
                        x = self.mixprop_convs[layer_idx](x, A_base)
                    if self.powermix_convs[layer_idx] is not None:
                        if len(new_supports) > 0:
                            A_base = new_supports[0]
                        else:
                            A_base = self._build_adaptive_adj()
                        x = self.powermix_convs[layer_idx](x, A_base)
                    elif self.gconv[layer_idx] is not None:
                        x = self.gconv[layer_idx](x, new_supports)
                    else:
                        x = self.residual_convs[layer_idx](x)
                else:
                    x = self.residual_convs[layer_idx](x)

                # 
                x = x + residual[:, :, :, -x.size(3):]
                x = self.bn[layer_idx](x)
                layer_idx += 1

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x



