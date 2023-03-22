import inspect
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch_scatter
from quantized_layers import MessagePassingQuant
from torch.nn import Module, ModuleDict, Parameter
from torch_scatter import scatter_add

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import (
    add_remaining_self_loops,
    add_self_loops,
    remove_self_loops,
    softmax,
)

REQUIRED_GCN_KEYS = [
    "weights",
    "inputs",
    "features",
    "norm",
]


class GCNConvQuant(MessagePassingQuant):
    def __init__(
        self,
        in_channels,
        out_channels,
        improved=False,
        cached=False,
        bias=True,
        normalize=True,
        mp_quantizers=None,
        layer_quantizers=None,
        **kwargs,
    ):
        super(GCNConvQuant,
              self).__init__(aggr="add", mp_quantizers=mp_quantizers, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        assert layer_quantizers is not None
        self.layer_quant_fns = layer_quantizers
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

        # create quantization modules for this layer
        self.layer_quantizers = ModuleDict()
        for key in REQUIRED_GCN_KEYS:
            self.layer_quantizers[key] = self.layer_quant_fns[key]()

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):

        # quantizing input
        x_q = self.layer_quantizers["inputs"](x)

        # quantizing layer weights
        w_q = self.layer_quantizers["weights"](self.weight)
        x = torch.matmul(x_q, w_q)

        x = self.layer_quantizers["features"](x)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    "Cached {} number of edges, but found {}. Please "
                    "disable the caching behavior of this layer by removing "
                    "the `cached=True` argument in its constructor.".format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(
                    edge_index,
                    x.size(self.node_dim),
                    edge_weight,
                    self.improved,
                    x.dtype,
                )
            else:
                norm = edge_weight

            norm = self.layer_quantizers["norm"](norm)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


REQUIRED_GAT_KEYS = [
    "weights",
    "inputs",
    "features",
    "attention",
    "alpha",
]


class GATConvQuant(MessagePassingQuant):
    def __init__(
        self,
        in_channels,
        out_channels,
        heads=1,
        concat=True,
        negative_slope=0.2,
        dropout=0,
        bias=True,
        mp_quantizers=None,
        layer_quantizers=None,
        **kwargs,
    ):
        super(GATConvQuant,
              self).__init__(aggr="add", mp_quantizers=mp_quantizers, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels,
                                             heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        assert layer_quantizers is not None
        self.layer_quant_fns = layer_quantizers
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

        # create quantization modules for this layer
        self.layer_quantizers = ModuleDict()
        for key in REQUIRED_GAT_KEYS:
            self.layer_quantizers[key] = self.layer_quant_fns[key]()

    def forward(self, x, edge_index, size=None):
        """"""

        # quantizing input
        x_q = self.layer_quantizers["inputs"](x)

        # quantizing layer weights
        w_q = self.layer_quantizers["weights"](self.weight)

        if size is None and torch.is_tensor(x_q):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x_q.size(0))

        if torch.is_tensor(x_q):
            x = torch.matmul(x_q, w_q)
            x_q = self.layer_quantizers["features"](x)
        else:
            x = (
                None if x_q[0] is None else torch.matmul(x_q[0], w_q),
                None if x_q[1] is None else torch.matmul(x_q[1], w_q),
            )

            x_q = (
                None
                if x[0] is None else self.layer_quantizers["features"](x[0]),
                None
                if x[1] is None else self.layer_quantizers["features"](x[1]),
            )

        return self.propagate(edge_index, size=size, x=x_q)

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_channels)
        att = self.layer_quantizers["attention"](self.att)

        if x_i is None:
            alpha = (x_j * att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * att).sum(dim=-1)

        alpha = self.layer_quantizers["alpha"](alpha)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return "{}({}, {}, heads={})".format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class GINConvQuant(MessagePassingQuant):
    """This is a reimplemented version of GINConv that uses the update function"""
    def __init__(self, nn, eps=0, train_eps=False, mp_quantizers=None,
                 **kwargs):
        super(GINConvQuant,
              self).__init__(aggr="add", mp_quantizers=mp_quantizers, **kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.nn.reset_parameters()
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, *args, **kwargs):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        out = self.propagate(edge_index, x=x)
        return out

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, x):
        return self.nn((1 + self.eps) * x + aggr_out)

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.nn)
