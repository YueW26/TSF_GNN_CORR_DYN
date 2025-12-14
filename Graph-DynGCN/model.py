import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 基础组件
# =========================

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


# =========================
# Chebyshev 谱卷积
# =========================

class ChebConv(nn.Module):
    """
    K阶Chebyshev谱域图卷积；保留alpha系数用于可解释性导出。
    """
    def __init__(self, c_in, c_out, K=3, dropout=0.0):
        super(ChebConv, self).__init__()
        assert K >= 1
        self.K = K
        self.dropout = dropout
        self.theta = nn.ModuleList(
            [nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=True) for _ in range(K)]
        )
        self.alpha = nn.Parameter(torch.ones(K), requires_grad=True)
        self.last_cheb_alphas = None  # 便于外部读取

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


# =========================
# 空间域 GCN（支持 diffusion / 幂律）
# =========================

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

        # 幂律系数（每阶一个），便于可解释
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
        with torch.no_grad():
            self.last_power_coef = self.power_coef.detach().cpu()

        out = [x]
        for A in supports:
            A_use = self._apply_diag_policy(A)
            if self.use_power: # power law
                # A^k 逐阶拼接
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
    """
    双图递推（固定邻接+自适应邻接）。
    """
    def __init__(self, c_in, c_out, K=3, droprate=0.1, temperature=1.0,
                 diag_mode='self_and_neighbor', r_dim=10, use_laplacian=False):
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

        self.r1 = nn.Parameter(torch.randn(1, r_dim), requires_grad=True)

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

    def _build_adj2_from_r1(self, N, device):
        r1 = self.r1
        if r1.size(0) != N:
            r1 = r1.expand(N, -1).contiguous()
        S = r1 @ r1.t()
        S = F.relu(S)
        adj_2 = torch.softmax(S / max(self.temperature, 1e-6), dim=1)
        adj_2 = self._apply_diag(adj_2, self.diag_mode)
        if self.training and self.droprate > 0:
            adj_2 = F.dropout(adj_2, p=self.droprate)
        return adj_2

    def forward(self, x, A_base):
        B, C, N, T = x.shape
        device = x.device

        adj_1 = self._build_adj1_from_A(A_base)
        adj_2 = self._build_adj2_from_r1(N, device)

        inj = [self.k_convs[k](x) * self.gates[k] for k in range(self.K)]

        # 路一
        z = inj[0].permute(0, 3, 2, 1).contiguous().view(B * T, N, -1)
        for k in range(1, self.K):
            z = adj_1 @ z + inj[k].permute(0, 3, 2, 1).contiguous().view(B * T, N, -1)
        z = z.view(B, T, N, -1).permute(0, 3, 2, 1).contiguous()

        # 路二
        z_fix = inj[0].permute(0, 3, 2, 1).contiguous().view(B * T, N, -1)
        for k in range(1, self.K):
            z_fix = adj_2 @ z_fix + inj[k].permute(0, 3, 2, 1).contiguous().view(B * T, N, -1)
        z_fix = z_fix.view(B, T, N, -1).permute(0, 3, 2, 1).contiguous()

        return z + z_fix



class PowerMixDual(nn.Module):
    """
    融合幂律传播和双图递推：
    - 双图 (固定图 + 自适应图)
    - 每阶传播带有 power-law 系数 alpha_k
    """
    def __init__(self, c_in, c_out, K=3, droprate=0.1, temperature=1.0,
                 diag_mode='self_and_neighbor', r_dim=10):
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

        self.r1 = nn.Parameter(torch.randn(1, r_dim), requires_grad=True)

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
        r1 = self.r1
        if r1.size(0) != N:
            r1 = r1.expand(N, -1).contiguous()
        S = r1 @ r1.t()
        return self._build_adj_softmax(S)
    
    def forward(self, x, A_base):
        B, C, N, T = x.shape
        device = x.device
        with torch.no_grad():
            self.last_power_coef = self.power_coef.detach().cpu()

        # === 从环境变量读取三类设计变体开关 ===
        use_random_base    = os.getenv("GWN_RANDOM_BASE_GRAPH", "0") == "1"   # D2: 随机基础图
        disable_second     = os.getenv("GWN_DISABLE_SECOND_GRAPH", "0") == "1" # D1: 只保留一路（single-graph）
        second_graph_fixed = os.getenv("GWN_SECOND_GRAPH_FIXED", "0") == "1"   # D2: 第二路固定（无自适应）

        # === 第一条路径使用的基础图（可随机） ===
        if use_random_base:
            A_work = torch.rand_like(A_base)
            # 可选：强制对称
            # A_work = 0.5 * (A_work + A_work.t())
        else:
            A_work = A_base

        adj_1 = self._build_adj1_from_A(A_work)

        # === 第二条路径：关 / 固定 / 自适应 ===
        if disable_second:
            adj_2 = None
        elif second_graph_fixed:
            adj_2 = self._build_adj1_from_A(A_work)
        else:
            adj_2 = self._build_adj2_from_r1(N, device)

        # === 注入 + 幂律递推 ===
        inj = [self.k_convs[k](x) * self.gates[k] for k in range(self.K)]

        # 路一
        z = inj[0].permute(0, 3, 2, 1).contiguous().view(B * T, N, -1)
        for k in range(1, self.K):
            coef = self.power_coef[k]
            z = adj_1 @ z + coef * inj[k].permute(0, 3, 2, 1).contiguous().view(B * T, N, -1)
        z = z.view(B, T, N, -1).permute(0, 3, 2, 1).contiguous()

        # 路二（可选）
        if adj_2 is None:
            z2 = 0
        else:
            z2 = inj[0].permute(0, 3, 2, 1).contiguous().view(B * T, N, -1)
            for k in range(1, self.K):
                coef = self.power_coef[k]
                z2 = adj_2 @ z2 + coef * inj[k].permute(0, 3, 2, 1).contiguous().view(B * T, N, -1)
            z2 = z2.view(B, T, N, -1).permute(0, 3, 2, 1).contiguous()

        return z + z2



# =========================
# Graph WaveNet 主体
# =========================

class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True,
                 addaptadj=True, aptinit=None, in_dim=2, out_dim=12,
                 residual_channels=32, dilation_channels=32, skip_channels=256,
                 end_channels=512, kernel_size=2, blocks=4, layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.device = device

        # 环境变量控制开关
        self.use_power = os.getenv("GWN_USE_POWER", "0") == "1"
        self.use_cheby = os.getenv("GWN_USE_CHEBY", "0") == "1"
        self.cheby_K = int(os.getenv("GWN_CHEBY_K", "3"))
        self.diag_mode = os.getenv("GWN_DIAG_MODE", "self_and_neighbor")

        # MixPropDual
        self.use_mixprop = os.getenv("GWN_USE_MIXPROP", "0") == "1"
        self.mixprop_K = int(os.getenv("GWN_MIXPROP_K", "3"))
        self.adj_droprate = float(os.getenv("GWN_ADJ_DROPOUT", "0.1"))
        self.adj_temperature = float(os.getenv("GWN_ADJ_TEMP", "1.0"))

        # PowerMixDual (NEW)
        self.use_powermix = os.getenv("GWN_USE_POWERMIX", "0") == "1"
        self.powermix_K = int(os.getenv("GWN_POWERMIX_K", "3"))
        self.powermix_droprate = float(os.getenv("GWN_POWERMIX_DROPOUT", "0.1"))
        self.powermix_temperature = float(os.getenv("GWN_POWERMIX_TEMP", "1.0"))

        # 模块容器
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

        # 计算感受野
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

        # 自适应邻接
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

        # 构建层
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

                # spatial模块选择
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

                # 空间聚合
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
                    elif self.powermix_convs[layer_idx] is not None:
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

                # 残差 & BN
                x = x + residual[:, :, :, -x.size(3):]
                x = self.bn[layer_idx](x)
                layer_idx += 1

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x


