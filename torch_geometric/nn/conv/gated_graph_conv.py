from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
from torch.nn import Parameter as Param
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing

from ..inits import uniform


class GatedGraphConv(MessagePassing):
    r"""The gated graph convolution operator from the `"Gated Graph Sequence
    Neural Networks" <https://arxiv.org/abs/1511.05493>`_ paper

    .. math::
        \mathbf{h}_i^{(0)} &= \mathbf{x}_i \, \Vert \, \mathbf{0}

        \mathbf{m}_i^{(l+1)} &= \sum_{j \in \mathcal{N}(i)} e_{j,i} \cdot
        \mathbf{\Theta} \cdot \mathbf{h}_j^{(l)}

        \mathbf{h}_i^{(l+1)} &= \textrm{GRU} (\mathbf{m}_i^{(l+1)},
        \mathbf{h}_i^{(l)})

    up to representation :math:`\mathbf{h}_i^{(L)}`.
    The number of input channels of :math:`\mathbf{x}_i` needs to be less or
    equal than :obj:`out_channels`.
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1`)

    Args:
        out_channels (int): Size of each input sample.
        num_layers (int): The sequence length :math:`L`.
        num_edge_types (int, optional): Number of edge types. (default: :obj:`1`)
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, out_channels: int, num_layers: int, num_edge_types: int = 1,
                 aggr: str = 'add', bias: bool = True, **kwargs):
        super(GatedGraphConv, self).__init__(aggr=aggr, **kwargs)

        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_edge_types = num_edge_types
        self.weight = Param(Tensor(num_edge_types, out_channels, out_channels))
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.out_channels, self.weight)
        self.rnn.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor=None,
                edge_weights: OptTensor = None) -> Tensor:
        if x.size(-1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if x.size(-1) < self.out_channels:
            zero = x.new_zeros(x.size(0), self.out_channels - x.size(-1))
            x = torch.cat([x, zero], dim=1)

        for _ in range(self.num_layers):
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            m = self.propagate(edge_index, x=x, edge_attr=edge_attr,
                                edge_weights=edge_weights, size=None)
            x = self.rnn(m, x)

        return x

    def message(self, x_j: Tensor, edge_attr: OptTensor=None,
                edge_weights: OptTensor=None):
        if edge_weights is not None:
            x_j = edge_weights.view(-1, 1) * x_j
        if self.weight.size(0) != 1 and edge_attr is not None:
            assert(self.weight.size(0) == edge_attr.size(1))
            edge_msgs = []
            for idx in range(self.num_edge_types):
                edge_attr_weight = self.weight[idx]
                edge_indices = edge_attr[:, idx]
                x_j_edge = x_j[edge_indices, :]
                x_j_edge = x_j @ edge_attr_weight
                edge_msgs.append(x_j_edge)
            x_j = torch.stack(edge_msgs, 2).sum(2)
        else:
            x_j = x_j @ self.weight[0]
        return x_j


    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, num_layers={}, num_edge_types={})'.format(
            self.__class__.__name__,
            self.out_channels,
            self.num_layers,
            self.num_edge_types
        )
