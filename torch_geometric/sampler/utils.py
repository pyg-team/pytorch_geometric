from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.storage import EdgeStorage
from torch_geometric.typing import EdgeType, OptTensor

# Since C++ cannot take dictionaries with tuples as key as input, edge type
# triplets need to be converted into single strings. This is done by adding
# EDGE_TYPE_SPLIT_STR between values in the edge type tuple:
EDGE_TYPE_SPLIT_STR = "__"

# Edge Type Conversion ########################################################


def edge_type_to_str(edge_type: Union[EdgeType, str]) -> str:
    if isinstance(edge_type, str):
        return edge_type
    return EDGE_TYPE_SPLIT_STR.join(edge_type)


def edge_type_to_str_dict(
        edge_type_dict: Dict[EdgeType, Any]) -> Dict[str, Any]:
    return {edge_type_to_str(k): v for k, v in edge_type_dict.items()}


def edge_type_from_str(edge_type: Union[EdgeType, str]) -> EdgeType:
    if isinstance(edge_type, tuple):
        return edge_type
    return tuple(edge_type.split(EDGE_TYPE_SPLIT_STR))


def edge_type_from_str_dict(
        edge_type_dict: Dict[str, Any]) -> Dict[EdgeType, Any]:
    return {edge_type_from_str(k): v for k, v in edge_type_dict.items()}


# Edge Layout Conversion ######################################################


# TODO deprecate when FeatureStore / GraphStore unification is complete
def to_csc(
    data: Union[Data, EdgeStorage],
    device: Optional[torch.device] = None,
    share_memory: bool = False,
    is_sorted: bool = False,
) -> Tuple[Tensor, Tensor, OptTensor]:
    # Convert the graph data into a suitable format for sampling (CSC format).
    # Returns the `colptr` and `row` indices of the graph, as well as an
    # `perm` vector that denotes the permutation of edges.
    # Since no permutation of edges is applied when using `SparseTensor`,
    # `perm` can be of type `None`.
    perm: Optional[Tensor] = None

    if hasattr(data, 'adj'):
        colptr, row, _ = data.adj.csc()

    elif hasattr(data, 'adj_t'):
        colptr, row, _ = data.adj_t.csr()

    elif data.edge_index is not None:
        (row, col) = data.edge_index
        if not is_sorted:
            perm = (col * data.size(0)).add_(row).argsort()
            row = row[perm]
        colptr = torch.ops.torch_sparse.ind2ptr(col[perm], data.size(1))
    else:
        row = torch.empty(0, dtype=torch.long, device=device)
        colptr = torch.zeros(data.num_nodes + 1, dtype=torch.long,
                             device=device)

    colptr = colptr.to(device)
    row = row.to(device)
    perm = perm.to(device) if perm is not None else None

    if not colptr.is_cuda and share_memory:
        colptr.share_memory_()
        row.share_memory_()
        if perm is not None:
            perm.share_memory_()

    return colptr, row, perm


def to_hetero_csc(
    data: HeteroData,
    device: Optional[torch.device] = None,
    share_memory: bool = False,
    is_sorted: bool = False,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, OptTensor]]:
    # Convert the heterogeneous graph data into a suitable format for sampling
    # (CSC format).
    # Returns dictionaries holding `colptr` and `row` indices as well as edge
    # permutations for each edge type, respectively.
    # Since C++ cannot take dictionaries with tuples as key as input, edge type
    # triplets are converted into single strings.
    colptr_dict, row_dict, perm_dict = {}, {}, {}

    for store in data.edge_stores:
        key = edge_type_to_str(store._key)
        out = to_csc(store, device, share_memory, is_sorted)
        colptr_dict[key], row_dict[key], perm_dict[key] = out

    return colptr_dict, row_dict, perm_dict
