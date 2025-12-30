import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import sympy
import scipy
import numpy as np
from torch.nn import init

'''
BWGNN model from "https://github.com/squareRoot3/Rethinking-Anomaly-Detection"
'''
class PolyConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(PolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias)
        self.lin = lin

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0]*feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k]*feat
        if self.lin:
            h = self.linear(h)
            h = self.activation(h)
        return h

def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d+1):
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d+1):
            inv_coeff.append(float(coeff[d-i]))
        thetas.append(inv_coeff)
    return thetas


class BWGNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, graph, d=2):
        super(BWGNN, self).__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.conv = nn.ModuleList()
        for i in range(len(self.thetas)):
            self.conv.append(PolyConv(h_feats, h_feats, self.thetas[i], lin=False))
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats*len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()
        self.d = d

    def forward(self, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([len(in_feat), 0], device=h.device)
        for conv in self.conv:
            h0 = conv(self.g, h)
            h_final = torch.cat([h_final, h0], -1)
        h = self.linear3(h_final)
        h = self.act(h)
        h = self.linear4(h)
        return h


class BWGNN_AFA(nn.Module):
    """
    BWGNN variant with Residual Frequency Attention (ResAFA).

    核心思路：
    1. 残差结构：输出 = 原始特征 + 注意力修正，保证性能下限即原模型表现；
    2. Tanh 掩码：可对频率成分进行增强或抑制；
    3. LayerNorm 稳定多层感知机的输入分布；
    4. 保持加权拼接的高维表达能力。
    """

    def __init__(self, in_feats, h_feats, num_classes, graph, d=2, dropout: float = 0.0):
        super().__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.conv = nn.ModuleList(
            [PolyConv(h_feats, h_feats, theta, lin=False) for theta in self.thetas]
        )
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.ln1 = nn.LayerNorm(h_feats)
        self.ln2 = nn.LayerNorm(h_feats)
        attn_hidden = max(1, h_feats // 2)
        self.attn_mlp = nn.Sequential(
            nn.Linear(h_feats, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
            nn.Tanh(),
        )
        self.linear3 = nn.Linear(h_feats * len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, in_feat):
        h = self.linear(in_feat)
        h = self.ln1(h)
        h = self.act(h)
        h = self.dropout(h)

        h = self.linear2(h)
        h = self.ln2(h)
        h = self.act(h)
        h = self.dropout(h)

        h_list = []
        for conv in self.conv:
            h_list.append(conv(self.g, h))

        h_stack = torch.stack(h_list, dim=1)
        n_nodes, c_orders, hid_dim = h_stack.shape
        attn_delta = self.attn_mlp(h_stack.view(-1, hid_dim)).view(n_nodes, c_orders, 1)
        h_enhanced = h_stack * (1.0 + attn_delta)
        h_flat = h_enhanced.view(n_nodes, -1)

        h = self.linear3(h_flat)
        h = self.act(h)
        h = self.dropout(h)
        h = self.linear4(h)
        return h


class BWGNN_Hetero_AFA(nn.Module):
    """
    Heterogeneous BWGNN with Residual Adaptive Frequency Attention (Concatenation Version).

    在关系内部采用加权拼接，保留所有频率阶数的信息，避免信息瓶颈。
    """

    def __init__(self, in_feats, h_feats, num_classes, graph, d=2, dropout: float = 0.0):
        super().__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)

        # 共享卷积参数
        self.conv = nn.ModuleList([PolyConv(h_feats, h_feats, theta, lin=False) for theta in self.thetas])

        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)

        # LayerNorm
        self.ln1 = nn.LayerNorm(h_feats)
        self.ln2 = nn.LayerNorm(h_feats)

        # Residual Attention MLP
        attn_hidden = max(1, h_feats // 2)
        self.attn_mlp = nn.Sequential(
            nn.Linear(h_feats, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
            nn.Tanh()
        )

        # 采用拼接，输入维度恢复为 h_feats * len(conv)
        self.linear3 = nn.Linear(h_feats * len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, in_feat):
        h_init = self.linear(in_feat)
        h_init = self.ln1(h_init)
        h_init = self.act(h_init)
        h_init = self.dropout(h_init)

        h_init = self.linear2(h_init)
        h_init = self.ln2(h_init)
        h_init = self.act(h_init)
        h_init = self.dropout(h_init)

        relation_outputs = []

        for relation in self.g.canonical_etypes:
            h_freq_list = []
            for conv in self.conv:
                h_freq_list.append(conv(self.g[relation], h_init))

            h_stack = torch.stack(h_freq_list, dim=1)
            n_nodes, c_orders, hid_dim = h_stack.shape

            attn_delta = self.attn_mlp(h_stack.view(-1, hid_dim)).view(n_nodes, c_orders, 1)
            h_enhanced = h_stack * (1.0 + attn_delta)

            # 拼接多阶特征
            h_rel_flat = h_enhanced.view(n_nodes, -1)
            h_rel = self.linear3(h_rel_flat)
            relation_outputs.append(h_rel)

        h_all = torch.stack(relation_outputs, dim=0).max(dim=0)[0]
        h_all = self.act(h_all)
        h_all = self.dropout(h_all)

        logits = self.linear4(h_all)
        return logits


# ============================================================================
# 消融实验变体：同构版 ResAFA
# ============================================================================

class BWGNN_AFA_NoResidual(nn.Module):
    """
    消融实验：去掉残差结构，直接使用注意力
    H_out = H_in * tanh(MLP(H_in))
    """
    def __init__(self, in_feats, h_feats, num_classes, graph, d=2, dropout: float = 0.0):
        super().__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.conv = nn.ModuleList(
            [PolyConv(h_feats, h_feats, theta, lin=False) for theta in self.thetas]
        )
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.ln1 = nn.LayerNorm(h_feats)
        self.ln2 = nn.LayerNorm(h_feats)
        attn_hidden = max(1, h_feats // 2)
        self.attn_mlp = nn.Sequential(
            nn.Linear(h_feats, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
            nn.Tanh(),
        )
        self.linear3 = nn.Linear(h_feats * len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, in_feat):
        h = self.linear(in_feat)
        h = self.ln1(h)
        h = self.act(h)
        h = self.dropout(h)

        h = self.linear2(h)
        h = self.ln2(h)
        h = self.act(h)
        h = self.dropout(h)

        h_list = []
        for conv in self.conv:
            h_list.append(conv(self.g, h))

        h_stack = torch.stack(h_list, dim=1)
        n_nodes, c_orders, hid_dim = h_stack.shape
        attn_delta = self.attn_mlp(h_stack.view(-1, hid_dim)).view(n_nodes, c_orders, 1)
        # 去掉残差：直接使用注意力，不使用 (1.0 + attn_delta)
        h_enhanced = h_stack * attn_delta
        h_flat = h_enhanced.view(n_nodes, -1)

        h = self.linear3(h_flat)
        h = self.act(h)
        h = self.dropout(h)
        h = self.linear4(h)
        return h


class BWGNN_AFA_NoLayerNorm(nn.Module):
    """
    消融实验：去掉 LayerNorm
    """
    def __init__(self, in_feats, h_feats, num_classes, graph, d=2, dropout: float = 0.0):
        super().__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.conv = nn.ModuleList(
            [PolyConv(h_feats, h_feats, theta, lin=False) for theta in self.thetas]
        )
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        # 去掉 LayerNorm
        attn_hidden = max(1, h_feats // 2)
        self.attn_mlp = nn.Sequential(
            nn.Linear(h_feats, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
            nn.Tanh(),
        )
        self.linear3 = nn.Linear(h_feats * len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, in_feat):
        h = self.linear(in_feat)
        # 去掉 LayerNorm
        h = self.act(h)
        h = self.dropout(h)

        h = self.linear2(h)
        # 去掉 LayerNorm
        h = self.act(h)
        h = self.dropout(h)

        h_list = []
        for conv in self.conv:
            h_list.append(conv(self.g, h))

        h_stack = torch.stack(h_list, dim=1)
        n_nodes, c_orders, hid_dim = h_stack.shape
        attn_delta = self.attn_mlp(h_stack.view(-1, hid_dim)).view(n_nodes, c_orders, 1)
        h_enhanced = h_stack * (1.0 + attn_delta)
        h_flat = h_enhanced.view(n_nodes, -1)

        h = self.linear3(h_flat)
        h = self.act(h)
        h = self.dropout(h)
        h = self.linear4(h)
        return h


class BWGNN_AFA_Sum(nn.Module):
    """
    消融实验：改用求和而不是拼接
    """
    def __init__(self, in_feats, h_feats, num_classes, graph, d=2, dropout: float = 0.0):
        super().__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.conv = nn.ModuleList(
            [PolyConv(h_feats, h_feats, theta, lin=False) for theta in self.thetas]
        )
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.ln1 = nn.LayerNorm(h_feats)
        self.ln2 = nn.LayerNorm(h_feats)
        attn_hidden = max(1, h_feats // 2)
        self.attn_mlp = nn.Sequential(
            nn.Linear(h_feats, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
            nn.Tanh(),
        )
        # 改用求和，输入维度为 h_feats
        self.linear3 = nn.Linear(h_feats, h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, in_feat):
        h = self.linear(in_feat)
        h = self.ln1(h)
        h = self.act(h)
        h = self.dropout(h)

        h = self.linear2(h)
        h = self.ln2(h)
        h = self.act(h)
        h = self.dropout(h)

        h_list = []
        for conv in self.conv:
            h_list.append(conv(self.g, h))

        h_stack = torch.stack(h_list, dim=1)
        n_nodes, c_orders, hid_dim = h_stack.shape
        attn_delta = self.attn_mlp(h_stack.view(-1, hid_dim)).view(n_nodes, c_orders, 1)
        h_enhanced = h_stack * (1.0 + attn_delta)
        # 改用求和而不是拼接
        h_sum = h_enhanced.sum(dim=1)

        h = self.linear3(h_sum)
        h = self.act(h)
        h = self.dropout(h)
        h = self.linear4(h)
        return h


# ============================================================================
# 消融实验变体：异构版 ResAFA
# ============================================================================

class BWGNN_Hetero_AFA_NoResidual(nn.Module):
    """
    消融实验：异构版去掉残差结构
    """
    def __init__(self, in_feats, h_feats, num_classes, graph, d=2, dropout: float = 0.0):
        super().__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.conv = nn.ModuleList([PolyConv(h_feats, h_feats, theta, lin=False) for theta in self.thetas])
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.ln1 = nn.LayerNorm(h_feats)
        self.ln2 = nn.LayerNorm(h_feats)
        attn_hidden = max(1, h_feats // 2)
        self.attn_mlp = nn.Sequential(
            nn.Linear(h_feats, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
            nn.Tanh()
        )
        self.linear3 = nn.Linear(h_feats * len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, in_feat):
        h_init = self.linear(in_feat)
        h_init = self.ln1(h_init)
        h_init = self.act(h_init)
        h_init = self.dropout(h_init)

        h_init = self.linear2(h_init)
        h_init = self.ln2(h_init)
        h_init = self.act(h_init)
        h_init = self.dropout(h_init)

        relation_outputs = []

        for relation in self.g.canonical_etypes:
            h_freq_list = []
            for conv in self.conv:
                h_freq_list.append(conv(self.g[relation], h_init))

            h_stack = torch.stack(h_freq_list, dim=1)
            n_nodes, c_orders, hid_dim = h_stack.shape

            attn_delta = self.attn_mlp(h_stack.view(-1, hid_dim)).view(n_nodes, c_orders, 1)
            # 去掉残差
            h_enhanced = h_stack * attn_delta

            h_rel_flat = h_enhanced.view(n_nodes, -1)
            h_rel = self.linear3(h_rel_flat)
            relation_outputs.append(h_rel)

        h_all = torch.stack(relation_outputs, dim=0).max(dim=0)[0]
        h_all = self.act(h_all)
        h_all = self.dropout(h_all)

        logits = self.linear4(h_all)
        return logits


class BWGNN_Hetero_AFA_NoLayerNorm(nn.Module):
    """
    消融实验：异构版去掉 LayerNorm
    """
    def __init__(self, in_feats, h_feats, num_classes, graph, d=2, dropout: float = 0.0):
        super().__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.conv = nn.ModuleList([PolyConv(h_feats, h_feats, theta, lin=False) for theta in self.thetas])
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        # 去掉 LayerNorm
        attn_hidden = max(1, h_feats // 2)
        self.attn_mlp = nn.Sequential(
            nn.Linear(h_feats, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
            nn.Tanh()
        )
        self.linear3 = nn.Linear(h_feats * len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, in_feat):
        h_init = self.linear(in_feat)
        # 去掉 LayerNorm
        h_init = self.act(h_init)
        h_init = self.dropout(h_init)

        h_init = self.linear2(h_init)
        # 去掉 LayerNorm
        h_init = self.act(h_init)
        h_init = self.dropout(h_init)

        relation_outputs = []

        for relation in self.g.canonical_etypes:
            h_freq_list = []
            for conv in self.conv:
                h_freq_list.append(conv(self.g[relation], h_init))

            h_stack = torch.stack(h_freq_list, dim=1)
            n_nodes, c_orders, hid_dim = h_stack.shape

            attn_delta = self.attn_mlp(h_stack.view(-1, hid_dim)).view(n_nodes, c_orders, 1)
            h_enhanced = h_stack * (1.0 + attn_delta)

            h_rel_flat = h_enhanced.view(n_nodes, -1)
            h_rel = self.linear3(h_rel_flat)
            relation_outputs.append(h_rel)

        h_all = torch.stack(relation_outputs, dim=0).max(dim=0)[0]
        h_all = self.act(h_all)
        h_all = self.dropout(h_all)

        logits = self.linear4(h_all)
        return logits


class BWGNN_Hetero_AFA_Sum(nn.Module):
    """
    消融实验：异构版改用求和而不是拼接
    """
    def __init__(self, in_feats, h_feats, num_classes, graph, d=2, dropout: float = 0.0):
        super().__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.conv = nn.ModuleList([PolyConv(h_feats, h_feats, theta, lin=False) for theta in self.thetas])
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.ln1 = nn.LayerNorm(h_feats)
        self.ln2 = nn.LayerNorm(h_feats)
        attn_hidden = max(1, h_feats // 2)
        self.attn_mlp = nn.Sequential(
            nn.Linear(h_feats, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
            nn.Tanh()
        )
        # 改用求和，输入维度为 h_feats
        self.linear3 = nn.Linear(h_feats, h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, in_feat):
        h_init = self.linear(in_feat)
        h_init = self.ln1(h_init)
        h_init = self.act(h_init)
        h_init = self.dropout(h_init)

        h_init = self.linear2(h_init)
        h_init = self.ln2(h_init)
        h_init = self.act(h_init)
        h_init = self.dropout(h_init)

        relation_outputs = []

        for relation in self.g.canonical_etypes:
            h_freq_list = []
            for conv in self.conv:
                h_freq_list.append(conv(self.g[relation], h_init))

            h_stack = torch.stack(h_freq_list, dim=1)
            n_nodes, c_orders, hid_dim = h_stack.shape

            attn_delta = self.attn_mlp(h_stack.view(-1, hid_dim)).view(n_nodes, c_orders, 1)
            h_enhanced = h_stack * (1.0 + attn_delta)
            # 改用求和而不是拼接
            h_sum = h_enhanced.sum(dim=1)

            h_rel = self.linear3(h_sum)
            relation_outputs.append(h_rel)

        h_all = torch.stack(relation_outputs, dim=0).max(dim=0)[0]
        h_all = self.act(h_all)
        h_all = self.dropout(h_all)

        logits = self.linear4(h_all)
        return logits


# heterogeneous graph
class BWGNN_Hetero(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, graph, d=2, dropout=0.0):
        """
        异构图的BWGNN模型
        
        修复说明：
        1. 修复了致命逻辑错误：不同关系应该并行处理原始特征，而不是串联
        2. 使用Max Pooling聚合关系（论文原实现）
        3. 使用ReLU激活函数（论文原实现）
        4. Dropout设为可选，默认0（与论文一致）
        
        Args:
            in_feats: 输入特征维度
            h_feats: 隐藏层维度
            num_classes: 类别数
            graph: DGL异构图
            d: Beta Wavelet的阶数
            dropout: Dropout概率（默认0.0，与论文一致，可设置为0.5用于正则化）
        """
        super(BWGNN_Hetero, self).__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.h_feats = h_feats
        # 为每个关系共享卷积参数
        self.conv = nn.ModuleList([PolyConv(h_feats, h_feats, theta, lin=False) for theta in self.thetas])
        
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats*len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        
        # 【修复】使用ReLU激活函数（论文原实现），而不是LeakyReLU
        self.act = nn.ReLU()
        # Dropout层（可选，默认0与论文一致）
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, in_feat):
        """
        前向传播
        
        修复前的问题：
        - 在循环中覆盖了h变量，导致关系2使用关系1的输出作为输入（错误！）
        
        修复后：
        - 每个关系都使用h_init（原始特征）作为输入
        - 最后聚合所有关系的结果
        """
        # 1. 初始特征变换
        h_init = self.linear(in_feat)
        h_init = self.dropout(self.act(h_init))
        h_init = self.linear2(h_init)
        h_init = self.dropout(self.act(h_init))
        
        h_all = []

        # 2. 并行处理每种关系
        # 【关键修复】每个关系都应该使用h_init作为输入，而不是上一个关系的输出
        for relation in self.g.canonical_etypes:
            h_final = torch.zeros([len(in_feat), 0], device=h_init.device)
            
            # 对当前关系应用多项式卷积
            # 【修正】这里必须始终使用h_init作为输入，绝不能覆盖它
            for conv in self.conv:
                # 获取当前关系的子图并应用卷积
                # 注意：dgl的heterograph索引方式
                h0 = conv(self.g[relation], h_init)
                h_final = torch.cat([h_final, h0], -1)
            
            # 聚合当前关系的多阶特征
            h_rel = self.linear3(h_final)
            h_all.append(h_rel)

        # 3. 【关键修复】聚合所有关系的结果：使用Max Pooling（论文原实现）
        # 论文："perform graph propagation for each relation separately and add a maximum pooling after that"
        # 修复前：h_all = torch.stack(h_all).sum(0)  # Sum Aggregation（错误）
        # 修复后：Max Pooling（正确，与论文一致）
        h_all = torch.stack(h_all, dim=0).max(dim=0)[0]
        h_all = self.act(h_all)
        h_all = self.dropout(h_all)  # 分类前再加一次dropout（如果dropout>0）
        
        # 4. 最终分类
        logits = self.linear4(h_all)
        return logits
