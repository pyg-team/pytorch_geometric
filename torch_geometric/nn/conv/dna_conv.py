import math
from typing import Optional, Tuple
from torch_geometric.typing import OptTensor

import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from ..inits import uniform, kaiming_uniform


class Linear(torch.nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, bias=True):
        super(Linear, self).__init__()
        assert in_channels % groups == 0 and out_channels % groups == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        self.weight = Parameter(
            torch.Tensor(groups, in_channels // groups,
                         out_channels // groups))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform(self.weight, fan=self.weight.size(1), a=math.sqrt(5))
        uniform(self.weight.size(1), self.bias)

    def forward(self, src):
        # Input: [*, in_channels]
        # Output: [*, out_channels]

        if self.groups > 1:
            size = src.size()[:-1]
            src = src.view(-1, self.groups, self.in_channels // self.groups)
            src = src.transpose(0, 1).contiguous()
            out = torch.matmul(src, self.weight)
            out = out.transpose(1, 0).contiguous()
            out = out.view(size + (self.out_channels, ))
        else:
            out = torch.matmul(src, self.weight.squeeze(0))

        if self.bias is not None:
            out += self.bias

        return out

    def __repr__(self):  # pragma: no cover
        return '{}({}, {}, groups={})'.format(self.__class__.__name__,
                                              self.in_channels,
                                              self.out_channels, self.groups)


def restricted_softmax(src, dim: int = -1, margin: float = 0.):
    src_max = torch.clamp(src.max(dim=dim, keepdim=True)[0], min=0.)
    out = (src - src_max).exp()
    out = out / (out.sum(dim=dim, keepdim=True) + (margin - src_max).exp())
    return out


class Attention(torch.nn.Module):
    def __init__(self, dropout=0):
        super(Attention, self).__init__()
        self.dropout = dropout

    def forward(self, query, key, value):
        return self.compute_attention(query, key, value)

    def compute_attention(self, query, key, value):
        # query: [*, query_entries, dim_k]
        # key: [*, key_entries, dim_k]
        # value: [*, key_entries, dim_v]
        # Output: [*, query_entries, dim_v]

        assert query.dim() == key.dim() == value.dim() >= 2
        assert query.size(-1) == key.size(-1)
        assert key.size(-2) == value.size(-2)

        # Score: [*, query_entries, key_entries]
        score = torch.matmul(query, key.transpose(-2, -1))
        score = score / math.sqrt(key.size(-1))
        score = restricted_softmax(score, dim=-1)
        score = F.dropout(score, p=self.dropout, training=self.training)

        return torch.matmul(score, value)

    def __repr__(self):  # pragma: no cover
        return '{}(dropout={})'.format(self.__class__.__name__, self.dropout)


class MultiHead(Attention):
    def __init__(self, in_channels, out_channels, heads=1, groups=1, dropout=0,
                 bias=True):
        super(MultiHead, self).__init__(dropout)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.groups = groups
        self.bias = bias

        assert in_channels % heads == 0 and out_channels % heads == 0
        assert in_channels % groups == 0 and out_channels % groups == 0
        assert max(groups, self.heads) % min(groups, self.heads) == 0

        self.lin_q = Linear(in_channels, out_channels, groups, bias)
        self.lin_k = Linear(in_channels, out_channels, groups, bias)
        self.lin_v = Linear(in_channels, out_channels, groups, bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_q.reset_parameters()
        self.lin_k.reset_parameters()
        self.lin_v.reset_parameters()

    def forward(self, query, key, value):
        # query: [*, query_entries, in_channels]
        # key: [*, key_entries, in_channels]
        # value: [*, key_entries, in_channels]
        # Output: [*, query_entries, out_channels]

        assert query.dim() == key.dim() == value.dim() >= 2
        assert query.size(-1) == key.size(-1) == value.size(-1)
        assert key.size(-2) == value.size(-2)

        query = self.lin_q(query)
        key = self.lin_k(key)
        value = self.lin_v(value)

        # query: [*, heads, query_entries, out_channels // heads]
        # key: [*, heads, key_entries, out_channels // heads]
        # value: [*, heads, key_entries, out_channels // heads]
        size = query.size()[:-2]
        out_channels_per_head = self.out_channels // self.heads

        query_size = size + (query.size(-2), self.heads, out_channels_per_head)
        query = query.view(query_size).transpose(-2, -3)

        key_size = size + (key.size(-2), self.heads, out_channels_per_head)
        key = key.view(key_size).transpose(-2, -3)

        value_size = size + (value.size(-2), self.heads, out_channels_per_head)
        value = value.view(value_size).transpose(-2, -3)

        # Output: [*, heads, query_entries, out_channels // heads]
        out = self.compute_attention(query, key, value)
        # Output: [*, query_entries, heads, out_channels // heads]
        out = out.transpose(-3, -2).contiguous()
        # Output: [*, query_entries, out_channels]
        out = out.view(size + (query.size(-2), self.out_channels))

        return out

    def __repr__(self):  # pragma: no cover
        return '{}({}, {}, heads={}, groups={}, dropout={}, bias={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.heads, self.groups, self.dropout, self.bias)


class DNAConv(MessagePassing):
    r"""The dynamic neighborhood aggregation operator from the `"Just Jump:
    Towards Dynamic Neighborhood Aggregation in Graph Neural Networks"
    <https://arxiv.org/abs/1904.04849>`_ paper

    .. math::
        \mathbf{x}_v^{(t)} = h_{\mathbf{\Theta}}^{(t)} \left( \mathbf{x}_{v
        \leftarrow v}^{(t)}, \left\{ \mathbf{x}_{v \leftarrow w}^{(t)} : w \in
        \mathcal{N}(v) \right\} \right)

    based on (multi-head) dot-product attention

    .. math::
        \mathbf{x}_{v \leftarrow w}^{(t)} = \textrm{Attention} \left(
        \mathbf{x}^{(t-1)}_v \, \mathbf{\Theta}_Q^{(t)}, [\mathbf{x}_w^{(1)},
        \ldots, \mathbf{x}_w^{(t-1)}] \, \mathbf{\Theta}_K^{(t)}, \,
        [\mathbf{x}_w^{(1)}, \ldots, \mathbf{x}_w^{(t-1)}] \,
        \mathbf{\Theta}_V^{(t)} \right)

    with :math:`\mathbf{\Theta}_Q^{(t)}, \mathbf{\Theta}_K^{(t)},
    \mathbf{\Theta}_V^{(t)}` denoting (grouped) projection matrices for query,
    key and value information, respectively.
    :math:`h^{(t)}_{\mathbf{\Theta}}` is implemented as a non-trainable
    version of :class:`torch_geometric.nn.conv.GCNConv`.

    .. note::
        In contrast to other layers, this operator expects node features as
        shape :obj:`[num_nodes, num_layers, channels]`.

    Args:
        channels (int): Size of each input/output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        groups (int, optional): Number of groups to use for all linear
            projections. (default: :obj:`1`)
        dropout (float, optional): Dropout probability of attention
            coefficients. (default: :obj:`0`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cache: Optional[Tuple[int, torch.Tensor, torch.Tensor]]

    def __init__(self, channels, heads=1, groups=1, dropout=0., cached=False,
                 bias=True, **kwargs):
        super(DNAConv, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.bias = bias
        self.cached = cached
        self._cache = None

        self.multi_head = MultiHead(channels, channels, heads, groups, dropout,
                                    bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.multi_head.reset_parameters()
        self._cache = None

    def __norm__(self, x, edge_index, edge_weight: OptTensor = None):

        cache = self._cache
        if cache is not None:
            if edge_index.size(1) != cache[0]:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please disable '
                    'the caching behavior of this layer by removing the '
                    '`cached=True` argument in its constructor.'.format(
                        cache[0], edge_index.size(1)))
            return cache[1:]

        num_edges = edge_index.size(1)

        edge_index, edge_weight = gcn_norm(edge_index, edge_weight,
                                           num_nodes=x.size(self.node_dim),
                                           dtype=x.dtype)

        if self.cached:
            self._cache = (num_edges, edge_index, edge_weight)

        return edge_index, edge_weight

    def forward(self, x, edge_index, edge_weight: OptTensor = None):
        """"""
        # x: [num_nodes, num_layers, channels]
        # edge_index: [2, num_edges]
        # edge_weight: [num_edges]

        if x.dim() != 3:
            raise ValueError('Feature shape must be [num_nodes, num_layers, '
                             'channels].')

        edge_index, norm = self.__norm__(x, edge_index, edge_weight)
        # propagate_type: (x: Tensor, norm: Tensor)
        return self.propagate(edge_index, x=x, norm=norm, size=None)

    def message(self, x_i, x_j, norm):
        # x_i: [num_edges, num_layers, channels]
        # x_j: [num_edges, num_layers, channels]
        # norm: [num_edges]
        # Output: [num_edges, channels]

        x_i = x_i[:, -1:]  # [num_edges, 1, channels]
        out = self.multi_head(x_i, x_j, x_j)  # [num_edges, 1, channels]
        return norm.view(-1, 1) * out.squeeze(1)

    def __repr__(self):
        return '{}({}, heads={}, groups={})'.format(
            self.__class__.__name__, self.multi_head.in_channels,
            self.multi_head.heads, self.multi_head.groups)
