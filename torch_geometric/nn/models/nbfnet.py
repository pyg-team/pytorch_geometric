from functools import reduce

import torch
from torch import LongTensor, Tensor
from torch.nn import Embedding, LayerNorm, Module, ModuleList, ReLU

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.models import MLP
from torch_geometric.typing import Optional, OptTensor, Tuple
from torch_geometric.utils import add_self_loops


def edge_match(edge_index, query_index):
    # source of algorithm:
    # https://github.com/KiddoZhu/NBFNet-PyG/blob/master/nbfnet/tasks.py#L6
    # we prepare a unique hashing of edges, mapping them to long ints
    # idea: max number of edges = num_nodes * num_relations
    # given a tuple (h, r), we search for all other existing edges
    # starting from head h with relation r
    base = edge_index.max(dim=1)[0] + 1
    # we make sure that the maximum product is less than MAX_LONG_INT
    assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max
    scale = base.cumprod(0)
    scale = torch.div(scale[-1], scale, rounding_mode='floor')
    # map both the original edge index and the query index to long ints
    edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)
    edge_hash, order = edge_hash.sort()
    query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)
    # matched ranges: [start[i], end[i])
    start = torch.bucketize(query_hash, edge_hash)
    end = torch.bucketize(query_hash, edge_hash, right=True)
    # num_match shows how many edges satisfy the (h, r) pattern
    # for each query in the batch
    num_match = end - start
    # generate the corresponding ranges
    offset = num_match.cumsum(0) - num_match
    range = torch.arange(num_match.sum(), device=edge_index.device)
    range = range + (start - offset).repeat_interleave(num_match)
    return order[range], num_match


def index_to_mask(index, size):
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask


class NBFConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('node_dim', 0)
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relation = Embedding(1, in_channels)
        self.linear = Linear(2 * in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.relation.reset_parameters()

    def forward(self, x: Tensor, boundary: Tensor, edge_index: LongTensor,
                edge_weight: OptTensor = None) -> Tensor:
        x0 = x.clone()
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        x = x + boundary
        x = torch.cat([x0, x], dim=-1)
        x = self.linear(x)
        return x

    def message(self, x_j: Tensor, edge_weight: OptTensor = None) -> Tensor:
        if edge_weight is None:
            return self.relation.weight * x_j
        else:
            return self.relation.weight * edge_weight.unsqueeze(-1).unsqueeze(
                -1) * x_j


class NBFModule(Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = NBFConv(in_channels, out_channels)
        self.norm = LayerNorm(out_channels)
        self.activation = ReLU()

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, x: Tensor, boundary: Tensor, edge_index: LongTensor,
                edge_weight: OptTensor = None) -> Tensor:
        x = self.conv(x, boundary, edge_index, edge_weight)
        x = self.norm(x)
        x = self.activation(x)
        return x


class NBFNet(Module):
    def __init__(self, in_channels: int, num_layers: int, hidden_channels: int,
                 residual: Optional[bool] = True,
                 symmetric: Optional[bool] = True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.query = Embedding(1, hidden_channels)
        self.layers = ModuleList([
            NBFModule(hidden_channels, hidden_channels)
            for _ in range(num_layers)
        ])
        self.mlp = MLP([2 * hidden_channels, 2 * hidden_channels, 1],
                       norm=None)
        self.residual = residual
        self.symmetric = symmetric

    def reset_parameters(self):
        self.query.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.mlp.reset_parameters()

    def remove_easy_edges(
            self, edge_label_index: LongTensor, edge_index: LongTensor,
            edge_weight: OptTensor = None) -> Tuple[LongTensor, OptTensor]:

        rev_edge_label_index = edge_label_index[[1, 0]]
        query_index = torch.cat([edge_label_index, rev_edge_label_index],
                                dim=1)
        match_index, _ = edge_match(edge_index, query_index)
        mask = ~index_to_mask(match_index, len(edge_index.T))
        edge_index = edge_index[:, mask]
        if edge_weight is not None:
            edge_weight = edge_weight[mask]
        return edge_index, edge_weight

    def forward(self, edge_label_index: LongTensor, edge_index: LongTensor,
                edge_weight: OptTensor = None) -> Tensor:

        num_nodes = edge_index.max().item() + 1
        batch_size = len(edge_label_index.T)

        if self.training:
            edge_index, edge_weight = self.remove_easy_edges(
                edge_label_index, edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes)

        r_index = torch.zeros(batch_size, dtype=torch.long,
                              device=edge_label_index.device)
        query = self.query(r_index)

        if self.symmetric:
            edge_label_index = (edge_label_index, edge_label_index[[1, 0]])
        else:
            edge_label_index = (edge_label_index)

        edge_feature = torch.zeros(batch_size, 1, 2 * self.hidden_channels,
                                   device=query.device)

        for h_index, t_index in edge_label_index:
            h_index = h_index.unsqueeze(-1)
            t_index = t_index.unsqueeze(-1)

            index = h_index.expand_as(query)
            boundary = torch.zeros(num_nodes, *query.shape,
                                   device=query.device)
            boundary.scatter_add_(dim=0, index=index.unsqueeze(0),
                                  src=query.unsqueeze(0))

            x = boundary.clone()
            for layer in self.layers:
                new_x = layer(x, boundary, edge_index, edge_weight)
                if self.residual:
                    new_x = new_x + x
                x = new_x

            x = torch.cat([x, query.expand(num_nodes, -1, -1)], dim=-1)
            x = x.transpose(0, 1)
            index = t_index.unsqueeze(-1).expand(-1, -1,
                                                 2 * self.hidden_channels)
            x = x.gather(1, index)
            edge_feature += x

        edge_feature = edge_feature / len(edge_label_index)
        score = self.mlp(edge_feature).squeeze(-1)
        return torch.sigmoid(score)
