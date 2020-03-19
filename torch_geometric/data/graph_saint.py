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

import time
import copy

import torch
from torch.multiprocessing import Pool
from torch_sparse import SparseTensor
from torch_sparse.saint import subgraph


class GraphSAINTSampler(object):
    def __init__(self, data, batch_size, num_steps=1, sample_coverage=25,
                 num_workers=0):
        assert data.edge_index is not None

        self.adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                                sparse_sizes=(data.num_nodes, data.num_nodes))

        self.data = copy.copy(data)
        self.data.edge_index = None

        self.batch_size = batch_size
        self.num_steps = num_steps
        self.sample_coverage = sample_coverage

        t = time.perf_counter()
        with Pool(10) as p:
            p.map(self.__sample__, [20 for _ in range(10)])
        print(time.perf_counter() - t)

        t = time.perf_counter()
        self.__sample__(200)
        print(time.perf_counter() - t)

    def __sample__(self, batch_size):
        raise NotImplementedError

    def __iter__(self):
        for _ in range(self.num_steps):
            yield self.__get_batch__()


class GraphSAINTNodeSampler(GraphSAINTSampler):
    def __sample__(self, num_examples):
        edge_sample = torch.randint(0, self.adj.nnz(),
                                    (num_examples, self.batch_size),
                                    dtype=torch.long)
        node_sample = self.adj.storage.row()[edge_sample]

        node_indices, edge_indices, adjs = [], [], []
        for sample in node_sample.sort(dim=1)[0].split(1, dim=0):
            node_idx = sample.unique_consecutive()
            adj, edge_index = subgraph(self.adj, node_idx)
            node_indices.append(node_idx)
            edge_indices.append(edge_indices)
            adjs.append(adj)

        return node_indices, edge_indices, adjs

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
