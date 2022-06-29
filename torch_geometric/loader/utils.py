import copy
import math
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch_sparse import SparseTensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.feature_store import FeatureStore
from torch_geometric.data.storage import EdgeStorage, NodeStorage
from torch_geometric.typing import EdgeType, OptTensor


def index_select(value: Tensor, index: Tensor, dim: int = 0) -> Tensor:
    out: Optional[Tensor] = None
    if torch.utils.data.get_worker_info() is not None:
        # If we are in a background process, we write directly into a shared
        # memory tensor to avoid an extra copy:
        size = list(value.size())
        size[dim] = index.numel()
        numel = math.prod(size)
        storage = value.storage()._new_shared(numel)
        out = value.new(storage).view(size)
    return torch.index_select(value, dim, index, out=out)


def edge_type_to_str(edge_type: Union[EdgeType, str]) -> str:
    # Since C++ cannot take dictionaries with tuples as key as input, edge type
    # triplets need to be converted into single strings.
    return edge_type if isinstance(edge_type, str) else '__'.join(edge_type)


def str_to_edge_type(key: Union[EdgeType, str]) -> EdgeType:
    return key if isinstance(key, tuple) else tuple(key.split('__'))


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

    elif hasattr(data, 'edge_index'):
        (row, col) = data.edge_index
        if not is_sorted:
            perm = (col * data.size(0)).add_(row).argsort()
            row = row[perm]
        colptr = torch.ops.torch_sparse.ind2ptr(col[perm], data.size(1))
    else:
        raise AttributeError("Data object does not contain attributes "
                             "'adj', 'adj_t' or 'edge_index'")

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


def filter_node_store_(store: NodeStorage, out_store: NodeStorage,
                       index: Tensor) -> NodeStorage:
    # Filters a node storage object to only hold the nodes in `index`:
    for key, value in store.items():
        if key == 'num_nodes':
            out_store.num_nodes = index.numel()

        elif store.is_node_attr(key):
            index = index.to(value.device)
            dim = store._parent().__cat_dim__(key, value, store)
            out_store[key] = index_select(value, index, dim=dim)

    return store


def filter_edge_store_(store: EdgeStorage, out_store: EdgeStorage, row: Tensor,
                       col: Tensor, index: Tensor,
                       perm: OptTensor = None) -> EdgeStorage:
    # Filters a edge storage object to only hold the edges in `index`,
    # which represents the new graph as denoted by `(row, col)`:
    for key, value in store.items():
        if key == 'edge_index':
            edge_index = torch.stack([row, col], dim=0)
            out_store.edge_index = edge_index.to(value.device)

        elif key == 'adj_t':
            # NOTE: We expect `(row, col)` to be sorted by `col` (CSC layout).
            row = row.to(value.device())
            col = col.to(value.device())
            edge_attr = value.storage.value()
            if edge_attr is not None:
                index = index.to(edge_attr.device)
                edge_attr = edge_attr[index]
            sparse_sizes = out_store.size()[::-1]
            # TODO Currently, we set `is_sorted=False`, see:
            # https://github.com/pyg-team/pytorch_geometric/issues/4346
            out_store.adj_t = SparseTensor(row=col, col=row, value=edge_attr,
                                           sparse_sizes=sparse_sizes,
                                           is_sorted=False, trust_data=True)

        elif store.is_edge_attr(key):
            dim = store._parent().__cat_dim__(key, value, store)
            if perm is None:
                index = index.to(value.device)
                out_store[key] = index_select(value, index, dim=dim)
            else:
                perm = perm.to(value.device)
                index = index.to(value.device)
                out_store[key] = index_select(value, perm[index], dim=dim)

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


def filter_feature_store(
    feature_store: FeatureStore,
    node_dict: Dict[str, Tensor],
    row_dict: Dict[str, Tensor],
    col_dict: Dict[str, Tensor],
    edge_dict: Dict[str, Tensor],
) -> HeteroData:
    r"""Constructs a `HeteroData` object from a feature store that only holds
    nodes in `node` end edges in `edge` for each node and edge type,
    respectively."""

    # Construct a new `HeteroData` object:
    data = HeteroData()

    # Filter edge storage:
    # TODO support edge attributes
    for key in edge_dict:
        edge_index = torch.stack([row_dict[key], col_dict[key]], dim=0)
        data[str_to_edge_type(key)].edge_index = edge_index

    # Filter node storage:
    attrs = feature_store.get_all_tensor_attrs()
    required_attrs = []
    for attr in attrs:
        if attr.group_name in node_dict:
            attr.index = node_dict[attr.group_name]
            required_attrs.append(attr)

    # NOTE Here, we utilize `feature_store.multi_get` to give the feature store
    # full control over optimizing how it returns features (since the call is
    # synchronous, this amounts to giving the feature store control over all
    # iteration).
    tensors = feature_store.multi_get_tensor(required_attrs)
    for i, attr in enumerate(required_attrs):
        data[attr.group_name][attr.attr_name] = tensors[i]

    return data
