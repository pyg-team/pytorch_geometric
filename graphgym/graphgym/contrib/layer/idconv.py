import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros, reset

from graphgym.config import cfg
from graphgym.register import register_layer


class GeneralIDConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, **kwargs):
        super(GeneralIDConvLayer, self).__init__(aggr=cfg.gnn.agg, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = cfg.gnn.normalize_adj

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_id = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.weight_id)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, id, edge_weight=None):
        """"""
        x_id = torch.index_select(x, dim=0, index=id)
        x_id = torch.matmul(x_id, self.weight_id)
        x = torch.matmul(x, self.weight)
        x.index_add_(0, id, x_id)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(self.node_dim),
                                             edge_weight, self.improved,
                                             x.dtype)
            else:
                norm = edge_weight
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
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GCNIDConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, normalize=True, **kwargs):
        super(GCNIDConvLayer, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_id = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.weight_id)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, id, edge_weight=None):
        """"""
        x_id = torch.index_select(x, dim=0, index=id)
        x_id = torch.matmul(x_id, self.weight_id)
        x = torch.matmul(x, self.weight)
        x.index_add_(0, id, x_id)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(self.node_dim),
                                             edge_weight, self.improved,
                                             x.dtype)
            else:
                norm = edge_weight
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
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class SAGEIDConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=False,
                 concat=False, bias=True, **kwargs):
        super(SAGEIDConvLayer, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.concat = concat

        in_channels = 2 * in_channels if concat else in_channels
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_id = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # uniform(self.weight.size(0), self.weight)
        # uniform(self.weight.size(0), self.weight_id)
        # uniform(self.weight.size(0), self.bias)
        glorot(self.weight)
        glorot(self.weight_id)
        zeros(self.bias)

    def forward(self, x, edge_index, id, edge_weight=None, size=None,
                res_n_id=None):
        """
        Args:
            res_n_id (Tensor, optional): Residual node indices coming from
                :obj:`DataFlow` generated by :obj:`NeighborSampler` are used to
                select central node features in :obj:`x`.
                Required if operating in a bipartite graph and :obj:`concat` is
                :obj:`True`. (default: :obj:`None`)
        """
        if not self.concat and torch.is_tensor(x):
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, 1, x.size(self.node_dim))

        return self.propagate(edge_index, size=size, x=x,
                              edge_weight=edge_weight, res_n_id=res_n_id, id=id)

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out, x, res_n_id, id):
        if self.concat and torch.is_tensor(x):
            aggr_out = torch.cat([x, aggr_out], dim=-1)
        elif self.concat and (isinstance(x, tuple) or isinstance(x, list)):
            assert res_n_id is not None
            aggr_out = torch.cat([x[0][res_n_id], aggr_out], dim=-1)

        aggr_out_id = torch.index_select(aggr_out, dim=0, index=id)
        aggr_out_id = torch.matmul(aggr_out_id, self.weight_id)
        aggr_out = torch.matmul(aggr_out, self.weight)
        aggr_out.index_add_(0, id, aggr_out_id)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)

        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GATIDConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(GATIDConvLayer, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels,
                                             heads * out_channels))
        self.weight_id = Parameter(torch.Tensor(in_channels,
                                                heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.weight_id)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, id, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index,
                                           num_nodes=x.size(self.node_dim))

        if torch.is_tensor(x):
            x_id = torch.index_select(x, dim=0, index=id)
            x_id = torch.matmul(x_id, self.weight_id)
            x = torch.matmul(x, self.weight)
            x.index_add_(0, id, x_id)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        return self.propagate(edge_index, size=size, x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

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
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class GINIDConvLayer(MessagePassing):
    def __init__(self, nn, nn_id, eps=0, train_eps=False, **kwargs):
        super(GINIDConvLayer, self).__init__(aggr='add', **kwargs)
        self.nn = nn
        self.nn_id = nn_id
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        reset(self.nn_id)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, id):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        x = (1 + self.eps) * x + self.propagate(edge_index, x=x)
        x_id = torch.index_select(x, dim=0, index=id)
        x_id = self.nn_id(x_id)
        x = self.nn(x)
        x.index_add_(0, id, x_id)
        return x

    def message(self, x_j):
        return x_j

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class GeneralIDConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GeneralIDConv, self).__init__()
        self.model = GeneralIDConvLayer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index,
                                        batch.node_id_index)
        return batch


class GCNIDConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GCNIDConv, self).__init__()
        self.model = GCNIDConvLayer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index,
                                        batch.node_id_index)
        return batch


class SAGEIDConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(SAGEIDConv, self).__init__()
        self.model = SAGEIDConvLayer(dim_in, dim_out, bias=bias, concat=True)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index,
                                        batch.node_id_index)
        return batch


class GATIDConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GATIDConv, self).__init__()
        self.model = GATIDConvLayer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index,
                                        batch.node_id_index)
        return batch


class GINIDConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GINIDConv, self).__init__()
        gin_nn = nn.Sequential(nn.Linear(dim_in, dim_out), nn.ReLU(),
                               nn.Linear(dim_out, dim_out))
        gin_nn_id = nn.Sequential(nn.Linear(dim_in, dim_out), nn.ReLU(),
                                  nn.Linear(dim_out, dim_out))
        self.model = GINIDConvLayer(gin_nn, gin_nn_id)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index,
                                        batch.node_id_index)
        return batch


register_layer('idconv', GeneralIDConv)
register_layer('gcnidconv', GCNIDConv)
register_layer('sageidconv', SAGEIDConv)
register_layer('gatidconv', GATIDConv)
register_layer('ginidconv', GINIDConv)
