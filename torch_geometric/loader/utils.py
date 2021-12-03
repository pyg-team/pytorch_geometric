from typing import Union, Tuple, Dict
from torch_geometric.typing import OptTensor, EdgeType

import copy

import torch
from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.storage import NodeStorage, EdgeStorage


def edge_type_to_str(edge_type: Union[EdgeType, str]) -> str:
    # Since C++ cannot take dictionaries with tuples as key as input, edge type
    # triplets need to be converted into single strings.
    return edge_type if isinstance(edge_type, str) else '__'.join(edge_type)


def to_csc(data: Union[Data, EdgeStorage]) -> Tuple[Tensor, Tensor, OptTensor]:
    # Convert the graph data into a suitable format for sampling (CSC format).
    # Returns the `colptr` and `row` indices of the graph, as well as an
    # `perm` vector that denotes the permutation of edges.
    # Since no permutation of edges is applied when using `SparseTensor`,
    # `perm` can be of type `None`.
    if hasattr(data, 'adj_t'):
        colptr, row, _ = data.adj_t.csr()
        return colptr, row, None

    elif hasattr(data, 'edge_index'):
        (row, col) = data.edge_index
        size = data.size()
        perm = (col * size[0]).add_(row).argsort()
        colptr = torch.ops.torch_sparse.ind2ptr(col[perm], size[1])
        return colptr, row[perm], perm

    raise AttributeError(
        "Data object does not contain attributes 'adj_t' or 'edge_index'")


def to_hetero_csc(
    data: HeteroData,
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
        colptr_dict[key], row_dict[key], perm_dict[key] = to_csc(store)

    return colptr_dict, row_dict, perm_dict


def filter_node_store_(store: NodeStorage, out_store: NodeStorage,
                       index: Tensor) -> NodeStorage:
    # Filters a node storage object to only hold the nodes in `index`:
    data = store._parent()

    for key, value in store.items():
        if key == 'num_nodes':
            out_store.num_nodes = index.numel()

        elif isinstance(data, Data) and data.is_node_attr(key):
            out_store[key] = value[index]

        elif isinstance(data, HeteroData) and isinstance(value, Tensor):
            out_store[key] = value[index]

    return store


def filter_edge_store_(store: EdgeStorage, out_store: EdgeStorage, row: Tensor,
                       col: Tensor, index: Tensor,
                       perm: OptTensor = None) -> EdgeStorage:
    # Filters a edge storage object to only hold the edges in `index`,
    # which represents the new graph as denoted by `(row, col)`:
    data = store._parent()

    for key, value in store.items():
        if key == 'edge_index':
            out_store.edge_index = torch.stack([row, col], dim=0)

        elif key == 'adj_t':
            # NOTE: We expect `(row, col)` to be sorted by `col` (CSC layout).
            edge_attr = value.storage.value()
            edge_attr = None if edge_attr is None else edge_attr[index]
            sparse_sizes = store.size()[::-1]
            out_store.adj_t = SparseTensor(row=col, col=row, value=edge_attr,
                                           sparse_sizes=sparse_sizes,
                                           is_sorted=True)

        elif isinstance(data, Data) and data.is_edge_attr(key):
            if perm is None:
                out_store[key] = value[index]
            else:
                out_store[key] = value[perm[index]]

        elif isinstance(data, HeteroData) and isinstance(value, Tensor):
            if perm is None:
                out_store[key] = value[index]
            else:
                out_store[key] = value[perm[index]]

    return store


def filter_data(data: Data, node: Tensor, row: Tensor, col: Tensor,
                edge: Tensor, perm: OptTensor = None) -> Data:
    # Filters a data object to only hold nodes in `node` and edges in `edge`:
    out = copy.copy(data)
    filter_node_store_(data._store, out._store, node)
    filter_edge_store_(data._store, out._store, row, col, edge, perm)
    return out


def filter_hetero_data(
    data: HeteroData,
    node_dict: Dict[str, Tensor],
    row_dict: Dict[str, Tensor],
    col_dict: Dict[str, Tensor],
    edge_dict: Dict[str, Tensor],
    perm_dict: Dict[str, OptTensor],
) -> HeteroData:
    # Filters a heterogeneous data object to only hold nodes in `node` and
    # edges in `edge` for each node and edge type, respectively:
    out = copy.copy(data)

    for node_type in data.node_types:
        filter_node_store_(data[node_type], out[node_type],
                           node_dict[node_type])

    for edge_type in data.edge_types:
        edge_type_str = edge_type_to_str(edge_type)
        filter_edge_store_(data[edge_type], out[edge_type],
                           row_dict[edge_type_str], col_dict[edge_type_str],
                           edge_dict[edge_type_str], perm_dict[edge_type_str])

    return out
