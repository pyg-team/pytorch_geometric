from typing import Callable, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.conv import EGNNConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import Adj, OptTensor


class EGNN(torch.nn.Module):
    r"""The edge-enhanced graph neural network from the
    `"Exploiting Edge Features in Graph Neural Networks"
    <https://arxiv.org/abs/1809.02709>`_ paper.

    Stacks :class:`~torch_geometric.nn.conv.EGNNConv` layers with either
    convolution (:obj:`backbone="gcn"`, EGNN(C)) or attention
    (:obj:`backbone="gat"`, EGNN(A)). Input edges are normalized once, and
    the attention variant passes evolved edge features between layers.
    Each layer concatenates the outputs of all edge channels. A final linear
    map produces node embeddings or logits of size :obj:`out_channels`.
    For graph prediction, apply a :pyg:`PyG` global pooling operator and a
    prediction head to these embeddings.

    Directed edges can be represented by three channels as in section 3.5
    of the paper: :math:`[A_{ij}, A_{ji}, A_{ij}+A_{ji}]`. For sparse inputs,
    concatenate the forward edges with features :math:`[w,0,w]` and reverse
    edges with features :math:`[0,w,w]`, then use
    :func:`~torch_geometric.utils.coalesce` to sum duplicate entries.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to infer
            it from the first input.
        hidden_channels (int): Hidden dimensionality **per edge channel**.
        out_channels (int): Size of each output sample.
        num_layers (int): Number of EGNN layers.
        edge_dim (int): Number of nonnegative edge feature channels.
        backbone (str, optional): :obj:`"gcn"` or :obj:`"gat"`.
            (default: :obj:`"gat"`)
        dropout (float, optional): Dropout on input and hidden node features
            and, for :obj:`"gat"`, normalized attention coefficients.
            (default: :obj:`0.0`)
        act (str or Callable, optional): Hidden-layer activation function.
            (default: :obj:`"elu"`)
        **kwargs (optional): Additional arguments of
            :class:`~torch_geometric.nn.conv.EGNNConv`, except
            :obj:`normalize`, which is managed by the model.

    Example:
        >>> import torch
        >>> from torch_geometric.nn import EGNN
        >>> model = EGNN(16, 32, 7, num_layers=2, edge_dim=3, backbone='gat')
        >>> x = torch.randn(4, 16)
        >>> edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        >>> edge_attr = torch.rand(4, 3)
        >>> model(x, edge_index, edge_attr).size()
        torch.Size([4, 7])
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        edge_dim: int,
        backbone: str = 'gat',
        dropout: float = 0.0,
        act: Union[str, Callable, None] = 'elu',
        **kwargs,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("'num_layers' must be positive")
        if out_channels < 1:
            raise ValueError("'out_channels' must be positive")
        if 'normalize' in kwargs:
            raise ValueError("Input normalization is managed by 'EGNN'")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.edge_dim = edge_dim
        self.backbone = backbone
        self.dropout = dropout
        self.act = activation_resolver(act)

        add_self_loops = kwargs.pop('add_self_loops', True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            channels = in_channels if i == 0 else edge_dim * hidden_channels
            self.convs.append(
                EGNNConv(channels, hidden_channels, edge_dim, backbone,
                         normalize=i == 0,
                         add_self_loops=add_self_loops if i == 0 else False,
                         dropout=dropout, **kwargs))
        self.lin = Linear(edge_dim * hidden_channels, out_channels)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()
        reset(self.act)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_edge_weights: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, Tensor]]]:
        r"""Computes node embeddings or logits, optionally returning the
        final evolved graph as COO indices and edge features.
        """
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x, (edge_index, edge_attr) = conv(x, edge_index, edge_attr,
                                              return_edge_weights=True)
            if i < self.num_layers - 1 and self.act is not None:
                x = self.act(x)
        x = self.lin(x)
        if return_edge_weights:
            return x, (edge_index, edge_attr)
        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers}, '
                f"backbone='{self.backbone}')")
