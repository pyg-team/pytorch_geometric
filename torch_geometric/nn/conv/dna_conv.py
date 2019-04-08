from __future__ import division

import math

import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing, GCNConv

from ..inits import uniform, kaiming_uniform


class Linear(torch.nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, bias=True):
        super(Linear, self).__init__()
        assert in_channels % groups == 0 and out_channels % groups == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        self.weight = Parameter(
            Tensor(groups, in_channels // groups, out_channels // groups))

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
            size = list(src.size())[:-1]
            src = src.view(-1, self.groups, self.in_channels // self.groups)
            src = src.transpose(0, 1).contiguous()
            out = torch.matmul(src, self.weight)
            out = out.transpose(1, 0).contiguous()
            out = out.view(*size, self.out_channels)
        else:
            out = torch.matmul(src, self.weight.squeeze(0))

        if self.bias is not None:
            out += self.bias

        return out

    def __repr__(self):
        return '{}({}, {}, groups={}, bias={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.groups, self.bias is not None)


def restricted_softmax(src, dim=-1, margin=0):
    src_max = torch.clamp(src.max(dim=dim, keepdim=True)[0], min=0)
    out = (src - src_max).exp()
    out = out / (out.sum(dim=dim, keepdim=True) + (margin - src_max).exp())
    return out


class Attention(torch.nn.Module):
    def __init__(self, dropout=0):
        super(Attention, self).__init__()
        self.dropout = dropout

    def forward(self, query, key, value):
        # Query: [*, query_entries, dim_k]
        # Key: [*, key_entries, dim_k]
        # Value: [*, key_entries, dim_v]
        # Output: [*, query_entries, dim_v]

        assert query.dim() == key.dim() == value.dim() >= 2
        assert query.size(-1) == key.size(-1)
        assert key.size(-2) == value.size(-2)

        # Score: [*, query_entries, key_entries]
        score = torch.matmul(query, key.transpose(-2, -1))
        score = score / math.sqrt(key.size(-1))
        score = restricted_softmax(score, dim=-1)

        if self.dropout > 0:
            score = F.dropout(score, p=self.dropout, training=self.training)

        return torch.matmul(score, value)

    def __repr__(self):
        return '{}(dropout={})'.format(self.__class__.__name__, self.dropout)


class MultiHead(Attention):
    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 groups=1,
                 dropout=0,
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
        # Query: [*, query_entries, in_channels]
        # Key: [*, key_entries, in_channels]
        # Value: [*, key_entries, in_channels]
        # Output: [*, query_entries, out_channels]

        assert query.dim() == key.dim() == value.dim() >= 2
        assert query.size(-1) == key.size(-1) == value.size(-1)
        assert key.size(-2) == value.size(-2)

        query = self.lin_q(query)
        key = self.lin_k(key)
        value = self.lin_v(value)

        # Query: [*, heads, query_entries, out_channels // heads]
        # Key: [*, heads, key_entries, out_channels // heads]
        # Value: [*, heads, key_entries, out_channels // heads]
        size = list(query.size())[:-2]
        out_channels_per_head = self.out_channels // self.heads

        query = query.view(*size, query.size(-2), self.heads,
                           out_channels_per_head).transpose(-2, -3)
        key = key.view(*size, key.size(-2), self.heads,
                       out_channels_per_head).transpose(-2, -3)
        value = value.view(*size, value.size(-2), self.heads,
                           out_channels_per_head).transpose(-2, -3)

        # Output: [*, heads, query_entries, out_channels // heads]
        out = super(MultiHead, self).forward(query, key, value)
        # Output: [*, query_entries, heads, out_channels // heads]
        out = out.transpose(-3, -2).contiguous()
        # Output: [*, query_entries, out_channels]
        out = out.view(*size, query.size(-2), self.out_channels)

        return out

    def __repr__(self):
        return '{}({}, {}, heads={}, groups={}, dropout={}, bias={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.heads, self.groups, self.dropout, self.bias)


class DNAConv(MessagePassing):
    def __init__(self,
                 channels,
                 heads=1,
                 groups=1,
                 dropout=0,
                 cached=False,
                 bias=True):
        super(DNAConv, self).__init__('add')
        self.bias = bias
        self.cached = cached
        self.cached_result = None
        self.multi_head = MultiHead(channels, channels, heads, groups, dropout,
                                    bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.multi_head.reset_parameters()
        self.cached_result = None

    def forward(self, x, edge_index):
        # X: [num_nodes, num_layers, channels]
        # EdgeIndex: [2, num_edges]

        if x.dim() != 3:
            raise ValueError('Feature shape must be [num_nodes, num_layers, '
                             'channels].')
        num_nodes, num_layers, channels = x.size()

        if not self.cached or self.cached_result is None:
            edge_index, norm = GCNConv.norm(
                edge_index, x.size(0), None, dtype=x.dtype)
            self.cached_result = edge_index, norm
        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_i, x_j, norm):
        # X_i: [num_edges, num_layers, channels]
        # X_j: [num_edges, num_layers, channels]
        # Norm: [num_edges]
        # Output: [num_edges, channels]

        x_i = x_i[:, -1:]  # [num_edges, 1, channels]
        out = self.multi_head(x_i, x_j, x_j)  # [num_edges, 1, channels]
        return norm.view(-1, 1) * out.squeeze(1)

    def __repr__(self):
        return '{}({}, heads={}, groups={})'.format(
            self.__class__.__name__, self.multi_head.in_channels,
            self.multi_head.heads, self.multi_head.groups)
