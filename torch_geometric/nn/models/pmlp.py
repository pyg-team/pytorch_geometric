import torch
import torch.nn.functional as F

from torch_geometric.nn import SimpleConv
from torch_geometric.nn.dense.linear import Linear


class PMLP(torch.nn.Module):
    r"""The PMLP model from the
        `"Graph Neural Networks are Inherently Good Generalizers:
        Insights by Bridging GNNs and MLPs"
        <https://arxiv.org/pdf/2212.09034.pdf>`_ paper where message passing
        layers are only used in inference


        Args:
            in_channels (int, optional): Size of each input sample.
            hidden_channels (int, optional): Size of each hidden sample.
            out_channels (int, optional): Size of each output sample.
                Will override :attr:`channel_list`. (default: :obj:`None`)
            num_layers (int, optional): The number of layers.
                Will override :attr:`channel_list`. (default: :obj:`None`)
            dropout (float, optional): Dropout probability of each
                hidden embedding.
            bias (bool, optional): If set to :obj:`False`, the module
                will not learn additive biases.
            **kwargs (optional): Extra parameters for the SimpleConv layers
                used during inference
        """
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.5,
                 bias: bool = True, **kwargs):
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
        self.simple_conv = SimpleConv(**kwargs)
        for param in self.simple_conv.parameters():
            param.requires_grad = False

        self.fcs = torch.nn.ModuleList(
            [Linear(in_channels, hidden_channels, bias=self.bias)])
        for _ in range(self.num_layers - 2):
            self.fcs.append(
                Linear(hidden_channels, hidden_channels, bias=self.bias))
        self.fcs.append(Linear(hidden_channels, out_channels, bias=self.bias))
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for mlp in self.fcs:
            torch.nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            torch.nn.init.zeros_(mlp.bias)

    def forward(self, x: torch.Tensor, edge_index=None) -> torch.Tensor:
        r"""
            Args:
            x (torch.Tensor): The source tensor.
            edge_index (torch.Tensor, optional):
        """
        for i in range(self.num_layers):
            x = x @ self.fcs[i].weight.t()
            if not self.training:
                x = self.simple_conv(x, edge_index=edge_index)
            if self.bias:
                x = x + self.fcs[i].bias
            if i != self.num_layers - 1:
                x = self.activation(self.bns(x))
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x
