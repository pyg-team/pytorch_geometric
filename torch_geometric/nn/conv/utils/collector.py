import inspect
from typing import Dict, Tuple, Optional, Any

import torch


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

        size = size = [None, None] if size is None else size
        size = size.tolist() if torch.is_tensor(size) else size
        size = list(size)
        assert len(size) == 2

        suffix2idx = self.conv.__suffix2id__()
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
                    index = edge_index[idx]
                    item = item.index_select(self.conv.node_dim, index)

            if item is inspect.Parameter.empty:
                continue

            out[key] = item

        # Assume homogeneous adjacency matrix :(
        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        out['edge_index_j'] = edge_index[suffix2idx['_j']]
        out['edge_index_i'] = edge_index[suffix2idx['_i']]
        out['index'] = out['edge_index_i']
        out['size_j'] = size[suffix2idx['_j']]
        out['size_i'] = size[suffix2idx['_i']]
        out['dim_size'] = out['size_i']

        return out
