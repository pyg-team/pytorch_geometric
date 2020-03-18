# Implement different sampling techniques
# Calculate p_{u,v} and p_v and return \alpha_{u,v} = p_{u,v} / p_v
# Calculate loss normalization \lambda_v = |V| * p_v
# p_v is the target node!

# For random node or random edge samplers, compute p_{u,v} and p_v analytically
# For other samplers, pre-generate and cache N sampled results:
# => Count number of occurrences C_v and C_{u,v} for each node and each edge.
# => Set \alpha_{u,v} to C_{u,v}/C_{v} and \lambda_v to C_v/N.

# Random node sampler:
# Sample nodes with node probability according to the in-degree of nodes.

# Random edge sampler:
# Sample edges with edge probability according to 1/deg(u) + 1/deg(v)

# Random walk sampler (multi-dimensional):
# r root nodes selected uniformly and each walker goes h hops
# Multi-dimensional random walk sampler as in:
# Ribeiro and Towsley - Estimating and sampling graphs with multidimensional
# random walks

import copy

import torch
from torch_scatter import scatter_add
from torch_sparse import SparseTensor


class GraphSAINTSampler(object):
    def __init__(self, data, batch_size, num_steps=1):
        assert data.edge_index is not None
        (row, col), edge_attr = data.edge_index, data.edge_attr

        size = (data.num_nodes, data.num_nodes)
        self.data = copy.copy(data)
        self.data.adj = SparseTensor(row=row, col=col, value=edge_attr,
                                     sparse_sizes=size)
        self.data.edge_index = None
        self.data.edge_attr = None

        self.batch_size = batch_size
        self.num_steps = num_steps

    def __get_batch__(self):
        raise NotImplementedError

    def __iter__(self):
        for _ in range(self.num_steps):
            yield self.__get_batch__()


class GraphSAINTNodeSampler(GraphSAINTSampler):
    def __init__(self, data, batch_size, num_steps=1):
        super(GraphSAINTNodeSampler, self).__init__(data, batch_size,
                                                    num_steps)

        adj = self.data.adj
        row, col, _ = adj.coo()

        inv_in_deg = adj.storage.colcount().to(torch.float).pow_(-1)
        inv_in_deg[inv_in_deg == float('inf')] = 0

        prob = inv_in_deg[col]  # TODO: row is way faster?
        prob.mul_(prob)

        prob = scatter_add(prob, row, dim=0, dim_size=adj.size(0))
        prob.div_(prob.sum())

        self.prob = prob

    def __get_batch__(self):
        node_idx = self.prob.multinomial(self.batch_size, replacement=True)
        node_idx = node_idx.unique()

        num_nodes = self.data.num_nodes
        data = copy.copy(self.data)

        adj = data.adj.permute(node_idx)
        data.adj = None

        for key, item in data:
            if item.size(0) == num_nodes:
                data[key] = item[node_idx]

        row, col, value = adj.coo()
        data.edge_index = torch.stack([row, col], dim=0)
        data.edge_attr = value

        return data
