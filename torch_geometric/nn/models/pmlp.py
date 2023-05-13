import torch
import torch.nn.functional as F

from torch_geometric.nn import SimpleConv
from torch_geometric.nn.dense.linear import Linear


class PMLP(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.5,
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.bias = bias
        self.bns = torch.nn.BatchNorm1d(hidden_channels, affine=False,
                                        track_running_stats=False)
        self.activation = torch.nn.ReLU()
        self.dropout = dropout

        self.fcs = torch.nn.ModuleList(
            [Linear(in_channels, hidden_channels, bias=self.bias)])
        for _ in range(self.num_layers - 2):
            self.fcs.append(
                Linear(hidden_channels, hidden_channels, bias=self.bias))
            self.fcs.append(
                Linear(hidden_channels, out_channels, bias=self.bias))
            self.reset_parameters()

    def reset_parameters(self):
        for mlp in self.fcs:
            torch.nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            torch.nn.init.zeros_(mlp.bias)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = x @ self.fcs[i].weight.t()
            if not self.training:
                x = SimpleConv(x, edge_index)
            if self.bias:
                x = x + self.fcs[i].bias
            if i != self.num_layers - 1:
                x = self.activation(self.bns(x))
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x
