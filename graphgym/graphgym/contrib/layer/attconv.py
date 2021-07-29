import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros
from graphgym.config import cfg
from graphgym.register import register_layer


class GeneralAddAttConvLayer(MessagePassing):
    r"""General GNN layer, with add attention"""

    def __init__(self, in_channels, out_channels,
                 improved=False, cached=False, bias=True, **kwargs):
        super(GeneralAddAttConvLayer, self).__init__(aggr=cfg.gnn.agg, **kwargs)

        self.heads = cfg.gnn.att_heads
        self.in_channels = int(in_channels // self.heads * self.heads)
        self.out_channels = int(out_channels // self.heads * self.heads)
        self.improved = improved
        self.cached = cached
        self.normalize = cfg.gnn.normalize_adj
        self.negative_slope = 0.2

        self.head_channels = out_channels // self.heads
        self.scaling = self.head_channels ** -0.5

        self.linear_msg = nn.Linear(in_channels, out_channels, bias=False)

        self.att = Parameter(
            torch.Tensor(1, self.heads, 2 * self.head_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
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

    def forward(self, x, edge_index, edge_weight=None):
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

        x = self.linear_msg(x)

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, edge_index_i, x_i, x_j, norm, size_i):
        x_i = x_i.view(-1, self.heads, self.head_channels)
        x_j = x_j.view(-1, self.heads, self.head_channels)
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        alpha = alpha.view(-1, self.heads, 1)
        return norm.view(-1,
                         1) * x_j * alpha if norm is not None else x_j * alpha

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.out_channels)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, {})'.format(self.__class__.__name__,
                                       self.in_channels,
                                       self.out_channels, self.heads)


class GeneralMulAttConvLayer(MessagePassing):
    r"""General GNN layer, with mul attention"""

    def __init__(self, in_channels, out_channels,
                 improved=False, cached=False, bias=True, **kwargs):
        super(GeneralMulAttConvLayer, self).__init__(aggr=cfg.gnn.agg, **kwargs)

        self.heads = cfg.gnn.att_heads
        self.in_channels = int(in_channels // self.heads * self.heads)
        self.out_channels = int(out_channels // self.heads * self.heads)
        self.improved = improved
        self.cached = cached
        self.normalize = cfg.gnn.normalize_adj
        self.negative_slope = 0.2

        self.head_channels = out_channels // self.heads
        self.scaling = self.head_channels ** -0.5

        self.linear_msg = nn.Linear(in_channels, out_channels, bias=False)

        # todo: curently only for single head attention
        # self.att = nn.Linear(out_channels, out_channels, bias=True)
        self.bias_att = Parameter(torch.Tensor(out_channels))
        self.scaler = torch.sqrt(torch.tensor(out_channels, dtype=torch.float))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.bias)
        zeros(self.bias_att)
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

    def forward(self, x, edge_index, edge_weight=None):
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

        x = self.linear_msg(x)

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, edge_index_i, x_i, x_j, norm, size_i):
        # todo: curently only for single head attention
        x_i = x_i.view(-1, self.heads, self.head_channels)
        x_j = x_j.view(-1, self.heads, self.head_channels)
        alpha = (x_i * x_j + self.bias_att).sum(dim=-1) / self.scaler
        # alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        alpha = alpha.view(-1, self.heads, 1)
        return norm.view(-1,
                         1) * x_j * alpha if norm is not None else x_j * alpha

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.out_channels)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, {})'.format(self.__class__.__name__,
                                       self.in_channels,
                                       self.out_channels, self.heads)


class GeneralAddAttConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GeneralAddAttConv, self).__init__()
        self.model = GeneralAddAttConvLayer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch


class GeneralMulAttConv(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GeneralMulAttConv, self).__init__()
        self.model = GeneralMulAttConvLayer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index)
        return batch


register_layer('gaddconv', GeneralAddAttConv)
register_layer('gmulconv', GeneralMulAttConv)


class GeneralEdgeAttConvv1Layer(MessagePassing):
    r"""Att conv with edge feature"""

    def __init__(self, in_channels, out_channels, task_channels=None,
                 improved=False, cached=False, bias=True, **kwargs):
        super(GeneralEdgeAttConvv1Layer, self).__init__(aggr=cfg.gnn.agg,
                                                        **kwargs)

        self.heads = cfg.gnn.att_heads
        self.in_channels = int(in_channels // self.heads * self.heads)
        self.out_channels = int(out_channels // self.heads * self.heads)
        self.task_channels = task_channels
        self.improved = improved
        self.cached = cached
        self.normalize = cfg.gnn.normalize_adj
        self.msg_direction = cfg.gnn.msg_direction
        self.negative_slope = 0.2

        self.head_channels = out_channels // self.heads
        self.scaling = self.head_channels ** -0.5

        if self.msg_direction == 'single':
            self.linear_msg = nn.Linear(in_channels + cfg.dataset.edge_dim,
                                        out_channels, bias=False)
        else:
            self.linear_msg = nn.Linear(in_channels * 2 + cfg.dataset.edge_dim,
                                        out_channels, bias=False)

        self.att_msg = Parameter(
            torch.Tensor(1, self.heads, self.head_channels))
        if self.task_channels is not None:
            self.att_task = Parameter(
                torch.Tensor(1, self.heads, self.task_channels))

        if cfg.gnn.att_final_linear:
            self.linear_final = nn.Linear(out_channels, out_channels,
                                          bias=False)
        if cfg.gnn.att_final_linear_bn:
            self.linear_final_bn = nn.BatchNorm1d(out_channels, eps=cfg.bn.eps,
                                                  momentum=cfg.bn.mom)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_msg)
        if self.task_channels is not None:
            glorot(self.att_task)
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

    def forward(self, x, edge_index, edge_weight=None, edge_feature=None,
                task_emb=None):
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

        return self.propagate(edge_index, x=x, norm=norm,
                              edge_feature=edge_feature, task_emb=task_emb)

    def message(self, edge_index_i, x_i, x_j, norm, size_i, edge_feature,
                task_emb):
        if self.msg_direction == 'both':
            x_j = torch.cat((x_i, x_j, edge_feature), dim=-1)
        else:
            x_j = torch.cat((x_j, edge_feature), dim=-1)
        x_j = self.linear_msg(x_j)
        x_j = x_j.view(-1, self.heads, self.head_channels)
        if task_emb is not None:
            task_emb = task_emb.view(1, 1, self.task_channels)
            alpha = (x_j * self.att_msg).sum(-1) + (
                        task_emb * self.att_task).sum(-1)
        else:
            alpha = (x_j * self.att_msg).sum(-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        alpha = alpha.view(-1, self.heads, 1)
        return norm.view(-1,
                         1) * x_j * alpha if norm is not None else x_j * alpha

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.out_channels)
        if cfg.gnn.att_final_linear_bn:
            aggr_out = self.linear_final_bn(aggr_out)
        if cfg.gnn.att_final_linear:
            aggr_out = self.linear_final(aggr_out)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, {})'.format(self.__class__.__name__,
                                       self.in_channels,
                                       self.out_channels, self.heads)


class GeneralEdgeAttConvv2Layer(MessagePassing):
    r"""Att conv with edge feature v2"""

    def __init__(self, in_channels, out_channels, task_channels=None,
                 improved=False, cached=False, bias=True, **kwargs):
        super(GeneralEdgeAttConvv2Layer, self).__init__(aggr=cfg.gnn.agg,
                                                        **kwargs)

        self.heads = cfg.gnn.att_heads
        self.in_channels = int(in_channels // self.heads * self.heads)
        self.out_channels = int(out_channels // self.heads * self.heads)
        self.task_channels = task_channels
        self.improved = improved
        self.cached = cached
        self.normalize = cfg.gnn.normalize_adj
        self.msg_direction = cfg.gnn.msg_direction
        self.negative_slope = 0.2

        self.head_channels = out_channels // self.heads
        self.scaling = self.head_channels ** -0.5

        if self.msg_direction == 'single':
            self.linear_value = nn.Linear(in_channels + cfg.dataset.edge_dim,
                                          out_channels, bias=bias)
            self.linear_key = nn.Linear(in_channels + cfg.dataset.edge_dim,
                                        out_channels, bias=bias)
        else:
            self.linear_value = nn.Linear(
                in_channels * 2 + cfg.dataset.edge_dim, out_channels, bias=bias)
            self.linear_key = nn.Linear(in_channels * 2 + cfg.dataset.edge_dim,
                                        out_channels, bias=bias)

        self.att_msg = Parameter(
            torch.Tensor(1, self.heads, self.head_channels))
        if self.task_channels is not None:
            self.att_task = Parameter(
                torch.Tensor(1, self.heads, self.task_channels))

        if cfg.gnn.att_final_linear:
            self.linear_final = nn.Linear(out_channels, out_channels,
                                          bias=False)
        if cfg.gnn.att_final_linear_bn:
            self.linear_final_bn = nn.BatchNorm1d(out_channels, eps=cfg.bn.eps,
                                                  momentum=cfg.bn.mom)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_msg)
        if self.task_channels is not None:
            glorot(self.att_task)
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

    def forward(self, x, edge_index, edge_weight=None, edge_feature=None,
                task_emb=None):
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

        if self.msg_direction == 'both':
            x = (x, x)  # todo: check if expected

        return self.propagate(edge_index, x=x, norm=norm,
                              edge_feature=edge_feature, task_emb=task_emb)

    def message(self, edge_index_i, x_i, x_j, norm, size_i, edge_feature,
                task_emb):
        if self.msg_direction == 'both':
            x_j = torch.cat((x_i, x_j, edge_feature), dim=-1)
        else:
            x_j = torch.cat((x_j, edge_feature), dim=-1)
        x_j = self.linear_value(x_j)
        x_j = x_j.view(-1, self.heads, self.head_channels)
        if task_emb is not None:
            task_emb = task_emb.view(1, 1, self.task_channels)
            alpha = (x_j * self.att_msg).sum(-1) + (
                        task_emb * self.att_task).sum(-1)
        else:
            alpha = (x_j * self.att_msg).sum(-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        alpha = alpha.view(-1, self.heads, 1)
        return norm.view(-1,
                         1) * x_j * alpha if norm is not None else x_j * alpha

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.out_channels)
        if cfg.gnn.att_final_linear_bn:
            aggr_out = self.linear_final_bn(aggr_out)
        if cfg.gnn.att_final_linear:
            aggr_out = self.linear_final(aggr_out)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, {})'.format(self.__class__.__name__,
                                       self.in_channels,
                                       self.out_channels, self.heads)


class GeneralEdgeAttConvv1(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GeneralEdgeAttConvv1, self).__init__()
        self.model = GeneralEdgeAttConvv1Layer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index,
                                        edge_feature=batch.edge_feature)
        return batch


class GeneralEdgeAttConvv2(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False, **kwargs):
        super(GeneralEdgeAttConvv2, self).__init__()
        self.model = GeneralEdgeAttConvv2Layer(dim_in, dim_out, bias=bias)

    def forward(self, batch):
        batch.node_feature = self.model(batch.node_feature, batch.edge_index,
                                        edge_feature=batch.edge_feature)
        return batch


register_layer('generaledgeattconvv1', GeneralEdgeAttConvv1)
register_layer('generaledgeattconvv2', GeneralEdgeAttConvv2)
