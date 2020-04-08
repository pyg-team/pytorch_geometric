from typing import Dict, Tuple, Any


class Collector(object):
    def bind(self, conv: Any):
        self.base_class = conv

    def collect(self, adj_type: Any, size: Tuple[int, int],
                kwargs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


##################################

# def __collect__(self, keys: List[str],
#                 kwargs: Dict[str, Any]) -> Dict[str, Any]:

#     out: Dict[str, Any] = {}
#     suffix2idx = self.__suffix2id__()
#     for key in keys:
#         if key[-2:] not in self.suffixes:
#             item = kwargs.get(key, inspect.Parameter.empty)
#         else:
#             item = kwargs.get(key[:-2], inspect.Parameter.empty)

#             if isinstance(item, (tuple, list)):
#                 assert len(item) == 2
#                 item = item[suffix2idx[key[-2:]]]

#         if item is inspect.Parameter.empty:
#             continue

#         out[key] = item

#     return out

# def __collect_for_edge_index__(self, edge_index: torch.Tensor,
#                                size: Optional[Tuple[int, int]],
#                                kwargs: Dict[str, Any]) -> Dict[str, Any]:
#     size = size = [None, None] if size is None else size
#     size = size.tolist() if torch.is_tensor(size) else size
#     size = list(size)
#     assert len(size) == 2

#     def update_size_(self, size, idx, item):
#         if torch.is_tensor(item) and size[idx] is None:
#             size[idx] = item.size(self.node_dim)
#         elif torch.is_tensor(item) and size[idx] != item.size(
#                 self.node_dim):
#             raise ValueError(
#                 (f'Encountered node tensor with size '
#                  f'{item.size(self.node_dim)} in dimension '
#                  f'{self.node_dim}, but expected size {size[idx]}.'))

#     keys = self.inspector.keys(['message', 'aggregate'])
#     out = self.__collect__(keys, kwargs)

#     suffix2idx = self.__suffix2id__()
#     for key, item in out.items():
#         if key[-2:] in self.suffixes:
#             idx = suffix2idx[key[-2:]]
#             update_size_(size, idx, item)
#             out[key] = item.index_select(self.node_dim, edge_index[idx])

#             tmp = kwargs[key[:-2]]
#             if isinstance(tmp, (tuple, list)):
#                 update_size_(size, 1 - idx, tmp[1 - idx])

#     size[0] = size[1] if size[0] is None else size[0]
#     size[1] = size[0] if size[1] is None else size[1]

#     out['edge_index_j'] = edge_index[suffix2idx['_j']]
#     out['edge_index_i'] = edge_index[suffix2idx['_i']]
#     out['index'] = out['edge_index_i']
#     out['size_j'] = size[suffix2idx['_j']]
#     out['size_i'] = size[suffix2idx['_i']]
#     out['dim_size'] = out['size_i']
