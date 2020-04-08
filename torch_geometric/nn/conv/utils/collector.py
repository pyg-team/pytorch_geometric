import inspect
from typing import Dict, Tuple, Optional, Any

import torch
from torch_scatter import gather_csr
from torch_sparse import SparseTensor


class Collector(object):
    def bind(self, conv: Any):
        self.conv = conv

    def collect(self, adj_type: Any, size: Optional[Tuple[int, int]],
                kwargs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class EdgeIndexSparseCollector(Collector):

    special_args = set([
        'edge_index_i', 'edge_index_j', 'size_i', 'size_j', 'index', 'dim_size'
    ])

    def update_size_(self, size, idx, item):
        node_dim = self.conv.node_dim
        if torch.is_tensor(item) and size[idx] is None:
            size[idx] = item.size(node_dim)
        elif torch.is_tensor(item) and size[idx] != item.size(node_dim):
            raise ValueError((f'Encountered node tensor with size '
                              f'{item.size(node_dim)} in dimension '
                              f'{node_dim}, but expected size {size[idx]}.'))

    def collect(self, edge_index: torch.Tensor,
                size: Optional[Tuple[int, int]],
                kwargs: Dict[str, Any]) -> Dict[str, Any]:

        size = [None, None] if size is None else size
        size = size.tolist() if torch.is_tensor(size) else size
        size = list(size)
        assert len(size) == 2

        suffix2idx: Dict[str, int] = {'_i': 0, '_j': 1}
        if self.conv.flow == 'target_to_source':
            suffix2idx = {'_i': 1, '_j': 0}
        keys = self.conv.inspector.keys(['message', 'aggregate'])

        out: Dict[str, Any] = {}
        for key in keys - self.special_args:
            if key[-2:] not in self.conv.suffixes:
                item = kwargs.get(key, inspect.Parameter.empty)
            else:
                idx = suffix2idx[key[-2:]]
                item = kwargs.get(key[:-2], inspect.Parameter.empty)

                if isinstance(item, (tuple, list)):
                    assert len(item) == 2
                    self.update_size_(size, 1 - idx, item[1 - idx])
                    item = item[idx]

                self.update_size_(size, idx, item)

                if torch.is_tensor(item):
                    index = edge_index[1 - idx]
                    item = item.index_select(self.conv.node_dim, index)

            if item is inspect.Parameter.empty:
                continue

            out[key] = item

        # Assume homogeneous adjacency matrix :(
        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        out['edge_index_j'] = edge_index[1 - suffix2idx['_j']]
        out['edge_index_i'] = edge_index[1 - suffix2idx['_i']]
        out['index'] = out['edge_index_i']
        out['size_j'] = size[suffix2idx['_j']]
        out['size_i'] = size[suffix2idx['_i']]
        out['dim_size'] = out['size_i']

        return out


class SparseAdjFusedCollector(Collector):

    special_args = set(['adj_t'])

    def collect(self, adj_t: SparseTensor, size: Optional[Tuple[int, int]],
                kwargs: Dict[str, Any]) -> Dict[str, Any]:

        suffix2idx: Dict[str, int] = {'_i': 0, '_j': 1}
        keys = self.conv.inspector.keys(['sparse_message_and_aggregate'])

        out: Dict[str, Any] = {}
        for key in keys - self.special_args:
            if key[-2:] not in self.conv.suffixes:
                item = kwargs.get(key, inspect.Parameter.empty)
            else:
                idx = suffix2idx[key[-2:]]
                item = kwargs.get(key[:-2], inspect.Parameter.empty)

                if isinstance(item, (tuple, list)):
                    assert len(item) == 2
                    item = item[idx]

            if item is inspect.Parameter.empty:
                continue

            out[key] = item

        out['adj_t'] = adj_t

        return out


class SparseAdjSparseCollector(Collector):

    special_args = set([
        'edge_index_i', 'edge_index_j', 'size_i', 'size_j', 'ptr', 'index',
        'dim_size'
    ])

    def collect(self, adj_t: SparseTensor, size: Optional[Tuple[int, int]],
                kwargs: Dict[str, Any]) -> Dict[str, Any]:

        suffix2idx: Dict[str, int] = {'_i': 0, '_j': 1}
        keys = self.conv.inspector.keys(['message', 'aggregate'])

        rowptr, col, _ = adj_t.csr()
        for _ in range(self.conv.node_dim):
            rowptr = rowptr.unsqueeze(0)

        out: Dict[str, Any] = {}
        for key in keys - self.special_args:
            if key[-2:] not in self.conv.suffixes:
                item = kwargs.get(key, inspect.Parameter.empty)
            else:
                idx = suffix2idx[key[-2:]]
                item = kwargs.get(key[:-2], inspect.Parameter.empty)

                if isinstance(item, (tuple, list)):
                    assert len(item) == 2
                    item = item[idx]

                if torch.is_tensor(item) and idx == 0:
                    item = gather_csr(item, rowptr)
                elif torch.is_tensor(item) and idx == 1:
                    item = item.index_select(self.conv.node_dim, col)

            if item is inspect.Parameter.empty:
                continue

            out[key] = item

        out['edge_index_j'] = col
        out['edge_index_i'] = adj_t.storage.row()
        out['index'] = out['edge_index_i']
        out['size_j'] = adj_t.sparse_size(0)
        out['size_i'] = adj_t.sparse_size(1)
        out['dim_size'] = out['size_i']
        out['ptr'] = rowptr

        return out


class DenseAdjFusedCollector(Collector):

    special_args = set(['adj_t'])

    def collect(self, adj_t: SparseTensor, size: Optional[Tuple[int, int]],
                kwargs: Dict[str, Any]) -> Dict[str, Any]:

        suffix2idx: Dict[str, int] = {'_i': 0, '_j': 1}
        keys = self.conv.inspector.keys(['dense_message_and_aggregate'])

        out: Dict[str, Any] = {}
        for key in keys - self.special_args:
            if key[-2:] not in self.conv.suffixes:
                item = kwargs.get(key, inspect.Parameter.empty)
            else:
                idx = suffix2idx[key[-2:]]
                item = kwargs.get(key[:-2], inspect.Parameter.empty)

                if isinstance(item, (tuple, list)):
                    assert len(item) == 2
                    item = item[idx]

            if item is inspect.Parameter.empty:
                continue

            out[key] = item

        out['adj_t'] = adj_t

        return out
