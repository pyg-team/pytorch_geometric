import torch
import torch.nn.functional as F
from torch_sparse import spspmm
from torch_geometric.nn import TopKPooling, GCNConv
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)


class GraphUNet(torch.nn.Module):
    def __init__(self, channels, depth, pool_ratio=0.8, sum_res=True,
                 act=F.relu):
        super(GraphUNet, self).__init__()

        self.channels = channels
        self.depth = depth
        self.pool_ratio = pool_ratio
        self.act = act
        self.sum_res = sum_res

        self.pools = torch.nn.ModuleList()
        self.down_convs = torch.nn.ModuleList()
        for i in range(depth):
            self.pools.append(TopKPooling(channels, pool_ratio))
            self.down_convs.append(GCNConv(channels, channels, improved=True))
        self.up_convs = torch.nn.ModuleList()
        for i in range(depth - 1):
            in_channels = channels if sum_res else 2 * channels
            self.up_convs.append(GCNConv(in_channels, channels, improved=True))

        self.reset_parameters()

    def reset_parameters(self):
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.down_convs:
            conv.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(self.depth):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i](
                x, edge_index, edge_weight, batch)
            x = self.act(self.down_convs[i](x, edge_index, edge_weight))

            if i < self.depth - 1:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            if i < self.depth - 1:
                x = self.act(self.up_convs[i](x, edge_index, edge_weight))

        return x

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def __repr__(self):
        return '{}({}, depth={}, pool_ratio={})'.format(
            self.__class__.__name__, self.channels, self.depth,
            self.pool_ratio)
