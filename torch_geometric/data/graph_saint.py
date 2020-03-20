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

import copy

import torch
from torch.multiprocessing import Queue, Process
from torch_sparse import SparseTensor
from torch_sparse.saint import subgraph
from torch_scatter import gather_csr


class GraphSAINTSampler(object):
    def __init__(self, data, batch_size, num_steps=1, sample_coverage=25,
                 num_workers=0):
        assert data.edge_index is not None
        assert 'aggr_norm' not in data
        assert 'loss_norm' not in data

        self.N = N = data.num_nodes
        self.E = data.num_edges

        self.adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                                value=data.edge_attr, sparse_sizes=(N, N))

        self.data = copy.copy(data)
        self.data.edge_index = None
        self.data.edge_attr = None

        self.batch_size = batch_size
        self.num_steps = num_steps
        self.sample_coverage = sample_coverage
        self.num_workers = num_workers

        if self.num_workers > 0:
            self.__sample_queue__ = Queue()
            self.__sample_workers__ = []
            for _ in range(self.num_workers):
                worker = Process(target=self.__put_sample__,
                                 args=(self.__sample_queue__, ))
                worker.daemon = True
                worker.start()
                self.__sample_workers__.append(worker)

        self.aggr_norm, self.loss_norm = self.__compute_norm__()

        if self.num_workers > 0:
            self.__data_queue__ = Queue()
            self.__data_workers__ = []
            for _ in range(self.num_workers):
                worker = Process(target=self.__put_data__,
                                 args=(self.__data_queue__, ))
                worker.daemon = True
                worker.start()
                self.__data_workers__.append(worker)

        self.__count__ = 0

    def __sample__(self):
        raise NotImplementedError

    def __compute_norm__(self):
        node_count = torch.zeros(self.N, dtype=torch.float)
        edge_count = torch.zeros(self.E, dtype=torch.float)

        samples, num_sampled_nodes = [], 0
        while num_sampled_nodes < self.N * self.sample_coverage:
            if self.num_workers > 0:
                for _ in range(200):
                    sample = self.__sample_queue__.get()
                    samples.append(sample)
                    num_sampled_nodes += sample[0].size(0)
            else:
                sample = self.__sample__(200)
                samples.extend(sample)
                num_sampled_nodes += sum([s[0].size(0) for s in sample])

        for sample in samples:
            node_idx, edge_idx, _ = sample
            node_count[node_idx] += 1
            edge_count[edge_idx] += 1

        rowptr = self.adj.storage.rowptr()
        aggr_norm = gather_csr(node_count, rowptr).div_(edge_count)
        aggr_norm.clamp_(0, 1e4)

        loss_norm = len(samples) / (node_count.clamp(1) * self.N)

        return aggr_norm, loss_norm

    def __get_data_from_sample__(self, sample):
        node_idx, edge_idx, adj = sample

        data = self.data.__class__()
        data.num_nodes = node_idx.size(0)
        row, col, value = adj.coo()
        data.edge_index = torch.stack([row, col], dim=0)
        data.edge_attr = value

        for key, item in self.data:
            if item.size(0) == self.N:
                data[key] = item[node_idx]
            elif item.size(0) == self.E:
                data[key] = item[edge_idx]
            else:
                data[key] = item

        data.aggr_norm = self.aggr_norm[edge_idx]
        data.loss_norm = self.loss_norm[node_idx]

        return data

    def __put_sample__(self, queue):
        while True:
            sample = self.__sample__(1)[0]
            queue.put(sample)

    def __put_data__(self, queue):
        while True:
            sample = self.__sample_queue__.get()
            data = self.__get_data_from_sample__(sample)
            queue.put(data)

    def __next__(self):
        if self.__count__ < len(self):
            self.__count__ += 1
            if self.num_workers > 0:
                data = self.__data_queue__.get()
            else:
                sample = self.__sample__(1)[0]
                data = self.__get_data_from_sample__(sample)
            return data
        else:
            raise StopIteration

    def __len__(self):
        return self.num_steps

    def __iter__(self):
        self.__count__ = 0
        return self


class GraphSAINTNodeSampler(GraphSAINTSampler):
    def __sample__(self, num_examples):
        edge_sample = torch.randint(0, self.adj.nnz(),
                                    (num_examples, self.batch_size),
                                    dtype=torch.long)
        node_sample = self.adj.storage.row()[edge_sample]

        samples = []
        for sample in node_sample.sort(dim=1)[0].split(1, dim=0):
            node_idx = sample.unique_consecutive()
            adj, edge_idx = subgraph(self.adj, node_idx)
            samples.append((node_idx, edge_idx, adj))
        return samples
