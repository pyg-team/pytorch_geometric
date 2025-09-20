from typing import Optional

import torch
from torch import Tensor

from torch_geometric.nn import Linear, MFConv, global_add_pool
from torch_geometric.typing import Adj


class NeuralFingerprint(torch.nn.Module):
    r"""The Neural Fingerprint model from the
    `"Convolutional Networks on Graphs for Learning Molecular Fingerprints"
    <https://arxiv.org/abs/1509.09292>`__ paper to generate fingerprints
    of molecules.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output fingerprint.
        num_layers (int): Number of layers.
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MFConv`.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers):
            in_channels = self.in_channels if i == 0 else self.hidden_channels
            self.convs.append(MFConv(in_channels, hidden_channels, **kwargs))

        self.lins = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.lins.append(Linear(hidden_channels, out_channels, bias=False))

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> Tensor:
        """"""  # noqa: D419
        outs = []
        for conv, lin in zip(self.convs, self.lins):
            x = conv(x, edge_index).sigmoid()
            y = lin(x).softmax(dim=-1)
            outs.append(global_add_pool(y, batch, batch_size))
        return sum(outs)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')
