from copy import copy
from typing import Dict, Optional, Union

import torch
from torch import Tensor

import torch_geometric
from torch_geometric.typing import EdgeType, NodeType, SparseTensor


@torch.jit._overload
def maybe_num_nodes(edge_index, num_nodes):
    # type: (Tensor, Optional[int]) -> int
    pass


@torch.jit._overload
def maybe_num_nodes(edge_index, num_nodes):
    # type: (SparseTensor, Optional[int]) -> int
    pass


def maybe_num_nodes(
    edge_index: Union[Tensor, SparseTensor],
    num_nodes: Optional[int] = None,
) -> int:
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        if torch_geometric.utils.is_torch_sparse_tensor(edge_index):
            return max(edge_index.size(0), edge_index.size(1))

        if torch.jit.is_tracing():
            # Avoid non-traceable if-check for empty `edge_index` tensor:
            tmp = torch.concat([
                edge_index.view(-1),
                edge_index.new_full((1, ), fill_value=-1)
            ])
            return tmp.max() + 1

        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))


def maybe_num_nodes_dict(
    edge_index_dict: Dict[EdgeType, Tensor],
    num_nodes_dict: Optional[Dict[NodeType, int]] = None,
) -> Dict[NodeType, int]:
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
