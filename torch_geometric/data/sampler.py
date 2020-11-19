import copy
from typing import List, Optional, Tuple, NamedTuple, Union

import torch
from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric.utils.num_nodes import maybe_num_nodes


class EdgeIndex(NamedTuple):
    edge_index: Tensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return EdgeIndex(edge_index, e_id, self.size)


class Adj(NamedTuple):
    adj_t: SparseTensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        adj_t = self.adj_t.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return Adj(adj_t, e_id, self.size)


class NeighborSampler(torch.utils.data.DataLoader):
    r"""The neighbor sampler from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, which allows
    for mini-batch training of GNNs on large-scale graphs where full-batch
    training is not feasible.

    Given a GNN with :math:`L` layers and a specific mini-batch of nodes
    :obj:`node_idx` for which we want to compute embeddings, this module
    iteratively samples neighbors and constructs bipartite graphs that simulate
    the actual computation flow of GNNs.

    More specifically, :obj:`sizes` denotes how much neighbors we want to
    sample for each node in each layer.
    This module then takes in these :obj:`sizes` and iteratively samples
    :obj:`sizes[l]` for each node involved in layer :obj:`l`.
    In the next layer, sampling is repeated for the union of nodes that were
    already encountered.
    The actual computation graphs are then returned in reverse-mode, meaning
    that we pass messages from a larger set of nodes to a smaller one, until we
    reach the nodes for which we originally wanted to compute embeddings.

    Hence, an item returned by :class:`NeighborSampler` holds the current
    :obj:`batch_size`, the IDs :obj:`n_id` of all nodes involved in the
    computation, and a list of bipartite graph objects via the tuple
    :obj:`(edge_index, e_id, size)`, where :obj:`edge_index` represents the
    bipartite edges between source and target nodes, :obj:`e_id` denotes the
    IDs of original edges in the full graph, and :obj:`size` holds the shape
    of the bipartite graph.
    For each bipartite graph, target nodes are also included at the beginning
    of the list of source nodes so that one can easily apply skip-connections
    or add self-loops.

    .. note::

        For an example of using :obj:`NeighborSampler`, see
        `examples/reddit.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        reddit.py>`_ or
        `examples/ogbn_products_sage.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        ogbn_products_sage.py>`_.

    Args:
        edge_index (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
            :obj:`torch_sparse.SparseTensor` that defines the underlying graph
            connectivity/message passing flow.
            :obj:`edge_index` holds the indices of a (sparse) symmetric
            adjacency matrix.
            If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its shape
            must be defined as :obj:`[2, num_edges]`, where messages from nodes
            :obj:`edge_index[0]` are sent to nodes in :obj:`edge_index[1]`
            (in case :obj:`flow="source_to_target"`).
            If :obj:`edge_index` is of type :obj:`torch_sparse.SparseTensor`,
            its sparse indices :obj:`(row, col)` should relate to
            :obj:`row = edge_index[1]` and :obj:`col = edge_index[0]`.
            The major difference between both formats is that we need to input
            the *transposed* sparse adjacency matrix.
        size ([int]): The number of neighbors to sample for each node in each
            layer. If set to :obj:`sizes[l] = -1`, all neighbors are included
            in layer :obj:`l`.
        node_idx (LongTensor, optional): The nodes that should be considered
            for creating mini-batches. If set to :obj:`None`, all nodes will be
            considered.
        num_nodes (int, optional): The number of nodes in the graph.
            (default: :obj:`None`)
        return_e_id (bool, optional): If set to :obj:`False`, will not return
            original edge indices of sampled edges. This is only useful in case
            when operating on graphs without edge features to save memory.
            (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
    def __init__(self, edge_index: Union[Tensor, SparseTensor],
                 sizes: List[int], node_idx: Optional[Tensor] = None,
                 num_nodes: Optional[int] = None, return_e_id: bool = True,
                 **kwargs):

        self.sizes = sizes
        self.return_e_id = return_e_id
        self.is_sparse_tensor = isinstance(edge_index, SparseTensor)
        self.__val__ = None

        # Obtain a *transposed* `SparseTensor` instance.
        edge_index = edge_index.to('cpu')
        if not self.is_sparse_tensor:
            num_nodes = maybe_num_nodes(edge_index, num_nodes)
            value = torch.arange(edge_index.size(1)) if return_e_id else None
            self.adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                      value=value,
                                      sparse_sizes=(num_nodes, num_nodes)).t()
        else:
            adj_t = edge_index
            if return_e_id:
                self.__val__ = adj_t.storage.value()
                value = torch.arange(adj_t.nnz())
                adj_t = adj_t.set_value(value, layout='coo')
            self.adj_t = adj_t

        self.adj_t.storage.rowptr()

        if node_idx is None:
            node_idx = torch.arange(self.adj_t.sparse_size(0))
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)

        super(NeighborSampler, self).__init__(
            node_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)

    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)

        adjs = []
        n_id = batch
        for size in self.sizes:
            adj_t, n_id = self.adj_t.sample_adj(n_id, size, replace=False)
            e_id = adj_t.storage.value()
            size = adj_t.sparse_sizes()[::-1]
            if self.__val__ is not None:
                adj_t.set_value_(self.__val__[e_id], layout='coo')

            if self.is_sparse_tensor:
                adjs.append(Adj(adj_t, e_id, size))
            else:
                row, col, _ = adj_t.coo()
                edge_index = torch.stack([col, row], dim=0)
                adjs.append(EdgeIndex(edge_index, e_id, size))

        if len(adjs) > 1:
            return batch_size, n_id, adjs[::-1]
        else:
            return batch_size, n_id, adjs[0]

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)


class RandomIndexSampler(torch.utils.data.Sampler):
    def __init__(self, num_nodes: int, num_parts: int, shuffle: bool = False):
        self.N = num_nodes
        self.num_parts = num_parts
        self.shuffle = shuffle
        self.n_ids = self.get_node_indices()

    def get_node_indices(self):
        n_id = torch.randint(self.num_parts, (self.N, ), dtype=torch.long)
        n_ids = [(n_id == i).nonzero(as_tuple=False).view(-1)
                 for i in range(self.num_parts)]
        return n_ids

    def __iter__(self):
        if self.shuffle:
            self.n_ids = self.get_node_indices()
        return iter(self.n_ids)

    def __len__(self):
        return self.num_parts


class RandomNodeSampler(torch.utils.data.DataLoader):
    r"""A data loader that randomly samples nodes within a graph and returns
    their induced subgraph.

    .. note::

        For an example of using :obj:`RandomNodeSampler`, see
        `examples/ogbn_proteins_deepgcn.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        ogbn_proteins_deepgcn.py>`_.

    Args:
        data (torch_geometric.data.Data): The graph data object.
        num_parts (int): The number of partitions.
        shuffle (bool, optional): If set to :obj:`True`, the data is reshuffled
            at every epoch (default: :obj:`False`).
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`num_workers`.
    """
    def __init__(self, data, num_parts: int, shuffle: bool = False, **kwargs):
        assert data.edge_index is not None

        self.N = N = data.num_nodes
        self.E = data.num_edges

        self.adj = SparseTensor(
            row=data.edge_index[0], col=data.edge_index[1],
            value=torch.arange(self.E, device=data.edge_index.device),
            sparse_sizes=(N, N))

        self.data = copy.copy(data)
        self.data.edge_index = None

        super(RandomNodeSampler, self).__init__(
            self, batch_size=1,
            sampler=RandomIndexSampler(self.N, num_parts, shuffle),
            collate_fn=self.__collate__, **kwargs)

    def __getitem__(self, idx):
        return idx

    def __collate__(self, node_idx):
        node_idx = node_idx[0]

        data = self.data.__class__()
        data.num_nodes = node_idx.size(0)
        adj, _ = self.adj.saint_subgraph(node_idx)
        row, col, edge_idx = adj.coo()
        data.edge_index = torch.stack([row, col], dim=0)

        for key, item in self.data:
            if isinstance(item, Tensor) and item.size(0) == self.N:
                data[key] = item[node_idx]
            elif isinstance(item, Tensor) and item.size(0) == self.E:
                data[key] = item[edge_idx]
            else:
                data[key] = item

        return data
