import copy

import torch
from torch.multiprocessing import Queue, Process
from torch_sparse import SparseTensor


class GraphSAINTSampler(object):
    def __init__(self, data, batch_size, num_steps=1, sample_coverage=25,
                 save_dir=None, num_workers=0):
        assert data.edge_index is not None
        assert 'node_norm' not in data
        assert 'edge_norm' not in data

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
        self.__count__ = 0

        # TODO: Add `save_dir` option with `tdqm` support.

        if self.num_workers > 0:
            self.__sample_queue__ = Queue()
            self.__sample_workers__ = []
            for _ in range(self.num_workers):
                worker = Process(target=self.__put_sample__,
                                 args=(self.__sample_queue__, ))
                worker.daemon = True
                worker.start()
                self.__sample_workers__.append(worker)

        self.node_norm, self.edge_norm = self.__compute_norm__()

        if self.num_workers > 0:
            self.__data_queue__ = Queue()
            self.__data_workers__ = []
            for _ in range(self.num_workers):
                worker = Process(target=self.__put_data__,
                                 args=(self.__data_queue__, ))
                worker.daemon = True
                worker.start()
                self.__data_workers__.append(worker)

    def __sample_nodes__(self, num_examples):
        raise NotImplementedError

    def __sample__(self, num_examples):
        node_samples = self.__sample_nodes__(num_examples)

        samples = []
        for node_idx in node_samples:
            node_idx = node_idx.unique()
            adj, edge_idx = self.adj.saint_subgraph(node_idx)
            samples.append((node_idx, edge_idx, adj))
        return samples

    def __compute_norm__(self):
        node_count = torch.zeros(self.N, dtype=torch.float)
        edge_count = torch.zeros(self.E, dtype=torch.float)

        num_samples = num_sampled_nodes = 0
        while num_sampled_nodes < self.N * self.sample_coverage:
            if self.num_workers > 0:
                for _ in range(200):
                    node_idx, edge_idx, _ = self.__sample_queue__.get()
                    node_count[node_idx] += 1
                    edge_count[edge_idx] += 1
                    num_samples += 1
                    num_sampled_nodes += node_idx.size(0)
            else:
                samples = self.__sample__(200)
                for node_idx, edge_idx, _ in samples:
                    node_count[node_idx] += 1
                    edge_count[edge_idx] += 1
                    num_samples += 1
                    num_sampled_nodes += node_idx.size(0)

        row, col, _ = self.adj.coo()

        edge_norm = (node_count[row] / edge_count).clamp_(0, 1e4)
        edge_norm[torch.isnan(edge_norm)] = 0.1

        node_count[node_count == 0] = 0.1
        node_norm = num_samples / node_count / self.N

        return node_norm, edge_norm

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

        data.node_norm = self.node_norm[node_idx]
        data.edge_norm = self.edge_norm[edge_idx]

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
    def __sample_nodes__(self, num_examples):
        edge_sample = torch.randint(0, self.E, (num_examples, self.batch_size),
                                    dtype=torch.long)
        node_sample = self.adj.storage.row()[edge_sample]
        return node_sample.unbind(dim=0)


class GraphSAINTEdgeSampler(GraphSAINTSampler):
    def __sample_nodes__(self, num_examples):
        # This function corresponds to the `Edge2` sampler of the official
        # implementation that weights all edges as equally important.
        # This is the default configuration in the GraphSAINT code.
        edge_sample = torch.randint(0, self.E, (num_examples, self.batch_size),
                                    dtype=torch.long)

        source_node_sample = self.adj.storage.row()[edge_sample]
        target_node_sample = self.adj.storage.col()[edge_sample]

        node_sample = torch.cat([source_node_sample, target_node_sample], -1)
        return node_sample.unbind(dim=0)


class GraphSAINTRandomWalkSampler(GraphSAINTSampler):
    def __init__(self, data, batch_size, walk_length, num_steps=1,
                 sample_coverage=25, save_dir=None, num_workers=0):
        self.walk_length = walk_length
        super(GraphSAINTRandomWalkSampler, self).__init__(
            data, batch_size, num_steps, sample_coverage, save_dir,
            num_workers)

    def __sample_nodes__(self, num_examples):
        start = torch.randint(0, self.N, (num_examples, self.batch_size),
                              dtype=torch.long)

        rowptr, col, _ = self.adj.csr()
        node_sample = torch.ops.torch_sparse.random_walk(
            rowptr, col, start.flatten(), self.walk_length)
        node_sample = node_sample.view(
            num_examples, self.batch_size * (self.walk_length + 1))
        return node_sample.unbind(dim=0)
