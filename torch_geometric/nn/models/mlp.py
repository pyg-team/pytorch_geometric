from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Identity

from torch_geometric.nn.dense.linear import Linear


class MLP(torch.nn.Module):
    r"""A Multi-Layer Perception (MLP) model.
    There exists two ways to instantiate an :class:`MLP`:

    1. By specifying explicit channel sizes, *e.g.*,

       .. code-block:: python

          mlp = MLP([16, 32, 64, 128])

       creates a three-layer MLP with **differently** sized hidden layers.

    1. By specifying fixed hidden channel sizes over a number of layers,
       *e.g.*,

       .. code-block:: python

          mlp = MLP(in_channels=16, hidden_channels=32,
                    out_channels=128, num_layers=3)

       creates a three-layer MLP with **equally** sized hidden layers.

    Args:
        channel_list (List[int] or int, optional): List of input, intermediate
            and output channels. :obj:`len(channel_list) - 1` denotes the
            number of layers of the MLP (default: :obj:`None`)
        in_channels (int, optional): Size of each input sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        hidden_channels (int, optional): Size of each hidden sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        out_channels (int, optional): Size of each output sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        num_layers (int, optional): The number of layers.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability of each hidden
            embedding. (default: :obj:`0.`)
        batch_norm (bool, optional): If set to :obj:`False`, will not make use
            of batch normalization. (default: :obj:`True`)
        relu_first (bool, optional): If set to :obj:`True`, ReLU activation is
            applied before batch normalization. (default: :obj:`False`)
    """
    def __init__(
        self,
        channel_list: Optional[Union[List[int], int]] = None,
        *,
        in_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout: float = 0.,
        batch_norm: bool = True,
        relu_first: bool = False,
    ):
        super().__init__()

        if not isinstance(channel_list, (tuple, list)):
            in_channels = channel_list

        if in_channels is not None:
            assert num_layers >= 1
            channel_list = [hidden_channels] * (num_layers - 1)
            channel_list = [in_channels] + channel_list + [out_channels]

        assert isinstance(channel_list, (tuple, list))
        assert len(channel_list) >= 2
        self.channel_list = channel_list
        self.dropout = dropout
        self.relu_first = relu_first

        self.lins = torch.nn.ModuleList()
        for dims in zip(channel_list[:-1], channel_list[1:]):
            self.lins.append(Linear(*dims))

        self.norms = torch.nn.ModuleList()
        for dim in zip(channel_list[1:-1]):
            self.norms.append(BatchNorm1d(dim) if batch_norm else Identity())

        self.reset_parameters()

    @property
    def in_channels(self) -> int:
        r"""Size of each input sample."""
        return self.channel_list[0]

    @property
    def out_channels(self) -> int:
        r"""Size of each output sample."""
        return self.channel_list[-1]

    @property
    def num_layers(self) -> int:
        r"""The number of layers."""
        return len(self.channel_list) - 1

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        """"""
        x = self.lins[0](x)
        for lin, norm in zip(self.lins[1:], self.norms):
            if self.relu_first:
                x = x.relu_()
            x = norm(x)
            if not self.relu_first:
                x = x.relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin.forward(x)
        return x

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self.channel_list)[1:-1]})'
