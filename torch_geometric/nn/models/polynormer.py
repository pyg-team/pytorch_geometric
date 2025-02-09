import torch
import torch.nn.functional as F

from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn.attention import PolynormerAttention


class Polynormer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        local_layers: int = 7,
        global_layers: int = 2,
        in_dropout: float = 0.15,
        dropout: float = 0.5,
        global_dropout: float = 0.5,
        heads: int = 1,
        beta: float = 0.9,
        pre_ln: bool = False,
        post_bn: bool = True,
        local_attn: bool = False,
    ):
        super().__init__()
        """
        _global = False
            Best Validation Accuracy: 71.90%
            Test Accuracy: 69.02%
        _global = True
            Best Validation Accuracy: 72.20%
            Test Accuracy: 70.05%
        """
        self._global = True
        self.in_drop = in_dropout
        self.dropout = dropout
        self.pre_ln = pre_ln
        self.post_bn = post_bn

        self.beta = beta

        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()
        if self.post_bn:
            self.post_bns = torch.nn.ModuleList()

        # first layer
        inner_channels = heads * hidden_channels
        self.h_lins.append(torch.nn.Linear(in_channels, inner_channels))
        if local_attn:
            self.local_convs.append(
                GATConv(in_channels, hidden_channels, heads=heads, concat=True,
                        add_self_loops=False, bias=False))
        else:
            self.local_convs.append(
                GCNConv(in_channels, inner_channels, cached=False,
                        normalize=True))

        self.lins.append(torch.nn.Linear(in_channels, inner_channels))
        self.lns.append(torch.nn.LayerNorm(inner_channels))
        if self.pre_ln:
            self.pre_lns.append(torch.nn.LayerNorm(in_channels))
        if self.post_bn:
            self.post_bns.append(torch.nn.BatchNorm1d(inner_channels))

        # following layers
        for _ in range(local_layers - 1):
            self.h_lins.append(torch.nn.Linear(inner_channels, inner_channels))
            if local_attn:
                self.local_convs.append(
                    GATConv(inner_channels, hidden_channels, heads=heads,
                            concat=True, add_self_loops=False, bias=False))
            else:
                self.local_convs.append(
                    GCNConv(inner_channels, inner_channels, cached=False,
                            normalize=True))

            self.lins.append(torch.nn.Linear(inner_channels, inner_channels))
            self.lns.append(torch.nn.LayerNorm(inner_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(heads *
                                                       hidden_channels))
            if self.post_bn:
                self.post_bns.append(torch.nn.BatchNorm1d(inner_channels))

        self.lin_in = torch.nn.Linear(in_channels, inner_channels)
        self.ln = torch.nn.LayerNorm(inner_channels)
        self.global_attn = torch.nn.Sequential(*[
            PolynormerAttention(
                hidden_channels,
                heads,
                hidden_channels,
                beta,
                global_dropout,
            ) for _ in range(global_layers)
        ])
        self.pred_local = torch.nn.Linear(inner_channels, out_channels)
        self.pred_global = torch.nn.Linear(inner_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for h_lin in self.h_lins:
            h_lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        if self.post_bn:
            for p_bn in self.post_bns:
                p_bn.reset_parameters()
        self.lin_in.reset_parameters()
        self.ln.reset_parameters()
        self.pred_local.reset_parameters()
        self.pred_global.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.in_drop, training=self.training)

        # equivariant local attention
        x_local = 0
        for i, local_conv in enumerate(self.local_convs):
            if self.pre_ln:
                x = self.pre_lns[i](x)
            h = self.h_lins[i](x)
            h = F.relu(h)
            x = local_conv(x, edge_index) + self.lins[i](x)
            if self.post_bn:
                x = self.post_bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = (1 - self.beta) * self.lns[i](h * x) + self.beta * x
            x_local = x_local + x

        # equivariant global attention
        if self._global:
            x_global = self.global_attn(self.ln(x_local))
            x = self.pred_global(x_global)
        else:
            x = self.pred_local(x_local)

        return x
