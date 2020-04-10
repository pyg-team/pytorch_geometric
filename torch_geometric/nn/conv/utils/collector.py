import inspect
from typing import Dict, Tuple, List, Optional, Any

import torch
from torch_scatter import gather_csr
from torch_sparse import SparseTensor

from .helpers import abs_dim, unsqueeze, unsqueeze_and_expand


class Collector(object):

    suffixes: List[str] = ['_i', '_j']

    def __init__(self, base_class: Any):
        self.base_class: Any = base_class

    def collect(self, adj_t: Any, size: Optional[Tuple[int, int]],
                kwargs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class SparseAdjFusedCollector(Collector):

    special_args = set(['adj_t'])

    def collect(self, adj_t: SparseTensor, size: Optional[Tuple[int, int]],
                kwargs: Dict[str, Any]) -> Dict[str, Any]:

        suffix2idx: Dict[str, int] = {'_i': 0, '_j': 1}
        keys = self.base_class.inspector.keys(['sparse_message_and_aggregate'])

        out: Dict[str, Any] = {}
        for key in keys - self.special_args:
            if key[-2:] not in self.suffixes:
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


class DenseAdjFusedCollector(Collector):

    special_args = set(['adj_t'])

    def collect(self, adj_t: torch.Tensor, size: Optional[Tuple[int, int]],
                kwargs: Dict[str, Any]) -> Dict[str, Any]:

        suffix2idx: Dict[str, int] = {'_i': 0, '_j': 1}
        keys = self.base_class.inspector.keys(['dense_message_and_aggregate'])

        out: Dict[str, Any] = {}
        for key in keys - self.special_args:
            if key[-2:] not in self.suffixes:
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


class EdgeIndexSparseCollector(Collector):

    special_args = set([
        'edge_index_i', 'edge_index_j', 'size_i', 'size_j', 'index', 'dim_size'
    ])

    def update_size_(self, size, idx, item):
        node_dim = self.base_class.node_dim
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
        if self.base_class.flow == 'target_to_source':
            suffix2idx = {'_i': 1, '_j': 0}
        keys = self.base_class.inspector.keys(['message', 'aggregate'])

        out: Dict[str, Any] = {}
        for key in keys - self.special_args:
            if key[-2:] not in self.suffixes:
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
                    item = item.index_select(self.base_class.node_dim, index)

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


class SparseAdjSparseCollector(Collector):

    special_args = set([
        'edge_index_i', 'edge_index_j', 'edge_attr', 'size_i', 'size_j', 'ptr',
        'index', 'dim_size'
    ])

    def collect(self, adj_t: SparseTensor, size: Optional[Tuple[int, int]],
                kwargs: Dict[str, Any]) -> Dict[str, Any]:

        node_dim = self.base_class.node_dim
        suffix2idx: Dict[str, int] = {'_i': 0, '_j': 1}
        keys = self.base_class.inspector.keys(['message', 'aggregate'])

        out: Dict[str, Any] = {}
        for key in keys - self.special_args:
            if key[-2:] not in self.suffixes:
                item = kwargs.get(key, inspect.Parameter.empty)
            else:
                idx = suffix2idx[key[-2:]]
                item = kwargs.get(key[:-2], inspect.Parameter.empty)

                if isinstance(item, (tuple, list)):
                    assert len(item) == 2
                    item = item[idx]

                if torch.is_tensor(item) and idx == 0:  # _i
                    ptr = adj_t.storage.rowptr()
                    ptr = unsqueeze(ptr, dim=0, length=abs_dim(item, node_dim))
                    item = gather_csr(item, ptr)

                elif torch.is_tensor(item) and idx == 1:  # _j
                    col = adj_t.storage.col()
                    item = item.index_select(self.base_class.node_dim, col)

            if item is inspect.Parameter.empty:
                continue

            out[key] = item

        out['edge_index_j'] = col
        out['edge_index_i'] = adj_t.storage.row()
        out['edge_attr'] = adj_t.storage.value()
        out['index'] = out['edge_index_i']
        out['size_j'] = adj_t.sparse_size(0)
        out['size_i'] = adj_t.sparse_size(1)
        out['dim_size'] = out['size_i']
        out['ptr'] = adj_t.storage.rowptr()

        return out


class SparseAdjPartialCollector(Collector):

    special_args = set(['edge_attr', 'edge_mask', 'inv_edge_mask'])

    def __init__(self, base_class: Any, fill_value: float = 0.,
                 max_deg: Optional[int] = None, binning: bool = True):
        super(SparseAdjPartialCollector, self).__init__(base_class)
        self.fill_value: float = fill_value
        self.max_deg: Optional[int] = max_deg
        self.binning: bool = binning

    def collect_message(self, adj_t: SparseTensor,
                        kwargs: Dict[str, Any]) -> Dict[str, Any]:

        node_dim = self.base_class.node_dim
        suffix2idx: Dict[str, int] = {'_i': 0, '_j': 1}
        keys = self.base_class.inspector.keys(['message'])

        out: Dict[str, Any] = {}
        for key in keys - self.special_args:
            if key[-2:] not in self.suffixes:
                item = kwargs.get(key, inspect.Parameter.empty)
            else:
                idx = suffix2idx[key[-2:]]
                item = kwargs.get(key[:-2], inspect.Parameter.empty)

                if isinstance(item, (tuple, list)):
                    assert len(item) == 2
                    item = item[idx]

                if torch.is_tensor(item) and idx == 0:  # _i
                    ptr = adj_t.storage.rowptr()
                    ptr = unsqueeze(ptr, dim=0, length=abs_dim(item, node_dim))
                    item = gather_csr(item, ptr)

                elif torch.is_tensor(item) and idx == 1:  # _j
                    col = adj_t.storage.col()
                    item = item.index_select(self.base_class.node_dim, col)

            if item is inspect.Parameter.empty:
                continue

            out[key] = item

        out['edge_attr'] = adj_t.storage.value()

        return out

    def collect_aggregate(self, adj_t: SparseTensor, inputs: torch.Tensor,
                          kwargs: Dict[str, Any],
                          message_kwargs: Dict[str, Any]) -> Dict[str, Any]:

        keys = self.base_class.inspector.keys(['partial_aggregate'])

        # TODO: Generate splits if not given.
        # Convert edge features, convert node features
        # Make to lists

        # CHECK IF USER WANTS TO ACCESS edge_attr, edge_mask, inv_edge_mask

        return inputs, kwargs


class DenseAdjPartialCollector(Collector):

    special_args = set(['edge_attr', 'edge_mask', 'inv_edge_mask'])

    def __init__(self, base_class: Any, fill_value: float = 0.):
        super(DenseAdjPartialCollector, self).__init__(base_class)
        self.fill_value: float = fill_value

    def collect_message(self, adj_t: torch.Tensor,
                        kwargs: Dict[str, Any]) -> Dict[str, Any]:

        node_dim = self.base_class.node_dim
        suffix2idx: Dict[str, int] = {'_i': 0, '_j': 1}
        keys = self.base_class.inspector.keys(['message', 'partial_aggregate'])
        keys = keys - self.special_args

        out: Dict[str, Any] = {}
        for key in keys - self.special_args:
            if key[-2:] not in self.suffixes:
                item = kwargs.get(key, inspect.Parameter.empty)
            else:
                idx = suffix2idx[key[-2:]]
                item = kwargs.get(key[:-2], inspect.Parameter.empty)

                if isinstance(item, (tuple, list)):
                    assert len(item) == 2
                    item = item[idx]

                # Convert to positive integer for correct unsqueezing.
                dim = abs_dim(item, node_dim)

                if torch.is_tensor(item) and idx == 0:  # _i
                    item = unsqueeze_and_expand(item, dim + 1, adj_t.size(dim))

                elif torch.is_tensor(item) and idx == 1:  # _j
                    item = unsqueeze_and_expand(item, dim, adj_t.size(dim + 1))

            if item is inspect.Parameter.empty:
                continue

            out[key] = item

        out['edge_attr'] = adj_t

        return out

    def collect_aggregate(self, adj_t: torch.Tensor, inputs: torch.Tensor,
                          kwargs: Dict[str, Any],
                          message_kwargs: Dict[str, Any]) -> Dict[str, Any]:

        keys = self.base_class.inspector.keys(['partial_aggregate'])

        # Find all entries zero entries. This is a bit tricker to implement
        # in case of multi-dimensional edge features where we aggregate the
        # features by taking the sum of their absolute values.
        dim = abs_dim(inputs, self.base_class.node_dim)
        if adj_t.dim() > dim + 2:
            dims = list(range(dim + 2, adj_t.dim()))
            inv_edge_mask = adj_t.abs().sum(dims) == 0
        else:
            inv_edge_mask = adj_t == 0

        message_kwargs['inv_edge_mask'] = 'inv_edge_mask'
        if 'edge_mask' in keys:
            message_kwargs['edge_mask'] = ~inv_edge_mask

        inv_edge_mask = unsqueeze(inv_edge_mask, dim=inv_edge_mask.dim(),
                                  length=inputs.dim() - inv_edge_mask.dim())
        inputs = inputs.contiguous()  # Needed for correct masking.
        inputs.masked_fill_(inv_edge_mask, self.fill_value)

        return inputs, message_kwargs
