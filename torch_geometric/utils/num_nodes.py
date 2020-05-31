from copy import copy
from typing import Optional

import torch


def maybe_num_nodes(index: torch.Tensor,
                    num_nodes: Optional[int] = None) -> int:
    return int(index.max()) + 1 if num_nodes is None else num_nodes


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
