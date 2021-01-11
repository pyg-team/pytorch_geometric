from copy import copy
from typing import Optional, Union

from torch import Tensor
from torch_sparse import SparseTensor


def maybe_num_nodes(edge_index: Union[Tensor, SparseTensor],
                    num_nodes: Optional[int] = None) -> int:
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1
    else:
        return max(edge_index.size(0), edge_index.size(1))


def maybe_num_nodes_dict(edge_index_dict, num_nodes_dict=None):
    num_nodes_dict = {} if num_nodes_dict is None else copy(num_nodes_dict)

    found_types = list(num_nodes_dict.keys())

    for keys, edge_index in edge_index_dict.items():

        key = keys[0]
        if key not in found_types:
            N = int(edge_index[0].max() + 1)
            num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

        key = keys[-1]
        if key not in found_types:
            N = int(edge_index[1].max() + 1)
            num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

    return num_nodes_dict
