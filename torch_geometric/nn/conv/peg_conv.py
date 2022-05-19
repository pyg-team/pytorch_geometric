import math

import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch_sparse import SparseTensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor, Size, Tensor
from torch_geometric.nn.inits import zeros, glorot


class PEGConv(MessagePassing):
    r"""The PEG layer from the
    `"Equivariant and Stable Positional Encoding
    for More Powerful Graph Neural Networks"
    <https://arxiv.org/abs/2203.00199>`_ paper


    Args:
        in_channels (int): Size of input node features.
        out_channels (int): Size of output node features.
        edge_mlp_dim (int):
            We use MLP to make one to one mapping between the
            relative information and edge weight. edge_mlp_dim
            represents the hidden units dimension in the MLP. (default: 32)
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)

        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 edge_mlp_dim: int = 32, improved: bool = False,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(PEGConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.edge_mlp_dim = edge_mlp_dim

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.edge_mlp = nn.Sequential(nn.Linear(1, edge_mlp_dim),
                                      nn.Linear(edge_mlp_dim, 1), nn.Sigmoid())

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, pos_encoding: Tensor,  edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        #coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        if isinstance(edge_index, Tensor):
            rel_coors = pos_encoding[edge_index[0]] - pos_encoding[edge_index[1]]
        elif isinstance(edge_index, SparseTensor):
            rel_coors = pos_encoding[
                edge_index.to_torch_sparse_coo_tensor()._indices()[0]] - pos_encoding[
                    edge_index.to_torch_sparse_coo_tensor()._indices()[1]]
        rel_dist = (rel_coors**2).sum(dim=-1, keepdim=True)

        
        # pos: l2 norms
        out = self.propagate(edge_index, x=x,
                                        edge_weight=edge_weight,
                                        pos=rel_dist, size=None)

        out = self.lin(out)
        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_weight: OptTensor,
                pos) -> Tensor:
        PE_edge_weight = self.edge_mlp(pos)
        return x_j if edge_weight is None else (PE_edge_weight *
                                                edge_weight.view(-1, 1) * x_j)

    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__,
                                     self.in_channels,
                                     self.out_channels)
