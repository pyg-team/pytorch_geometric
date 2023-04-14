import copy
import math
from collections.abc import Sequence
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from torch_geometric.data import (
    Data,
    FeatureStore,
    GraphStore,
    HeteroData,
    TensorAttr,
    remote_backend_utils,
)
from torch_geometric.data.storage import EdgeStorage, NodeStorage
from torch_geometric.typing import (
    EdgeType,
    FeatureTensorType,
    InputEdges,
    InputNodes,
    NodeType,
    OptTensor,
    SparseTensor,
)


def index_select(value: FeatureTensorType, index: Tensor,
                 dim: int = 0) -> Tensor:

    # PyTorch currently only supports indexing via `torch.int64` :(
    index = index.to(torch.int64)

    if isinstance(value, Tensor):
        out: Optional[Tensor] = None
        if torch.utils.data.get_worker_info() is not None:
            # If we are in a background process, we write directly into a
            # shared memory tensor to avoid an extra copy:
            size = list(value.shape)
            size[dim] = index.numel()
            numel = math.prod(size)
            storage = value.storage()._new_shared(numel)
            out = value.new(storage).view(size)

        return torch.index_select(value, dim, index, out=out)

    elif isinstance(value, np.ndarray):
        return torch.from_numpy(np.take(value, index, axis=dim))

    raise ValueError(f"Encountered invalid feature tensor type "
                     f"(got '{type(value)}')")


def filter_node_store_(store: NodeStorage, out_store: NodeStorage,
                       index: Tensor):
    # Filters a node storage object to only hold the nodes in `index`:
    for key, value in store.items():
        if key == 'num_nodes':
            out_store.num_nodes = index.numel()

        elif store.is_node_attr(key):
            if isinstance(value, Tensor):
                index = index.to(value.device)
            elif isinstance(value, np.ndarray):
                index = index.cpu()
            dim = store._parent().__cat_dim__(key, value, store)
            out_store[key] = index_select(value, index, dim=dim)


def filter_edge_store_(store: EdgeStorage, out_store: EdgeStorage, row: Tensor,
                       col: Tensor, index: Tensor, perm: OptTensor = None):
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
                edge_attr = index_select(edge_attr, index, dim=0)
            sparse_sizes = out_store.size()[::-1]
            # TODO Currently, we set `is_sorted=False`, see:
            # https://github.com/pyg-team/pytorch_geometric/issues/4346
            out_store.adj_t = SparseTensor(row=col, col=row, value=edge_attr,
                                           sparse_sizes=sparse_sizes,
                                           is_sorted=False, trust_data=True)

        elif store.is_edge_attr(key):
            dim = store._parent().__cat_dim__(key, value, store)
            if isinstance(value, Tensor):
                index = index.to(value.device)
            elif isinstance(value, np.ndarray):
                index = index.cpu()
            if perm is None:
                out_store[key] = index_select(value, index, dim=dim)
            else:
                if isinstance(value, Tensor):
                    perm = perm.to(value.device)
                elif isinstance(value, np.ndarray):
                    perm = perm.cpu()
                out_store[key] = index_select(
                    value,
                    perm[index.to(torch.int64)],
                    dim=dim,
                )


def filter_data(data: Data, node: Tensor, row: Tensor, col: Tensor,
                edge: Tensor, perm: OptTensor = None) -> Data:
    # Filters a data object to only hold nodes in `node` and edges in `edge`:
    out = copy.copy(data)
    filter_node_store_(data._store, out._store, node)
    filter_edge_store_(data._store, out._store, row, col, edge, perm)
    return out


def filter_hetero_data(
    data: HeteroData,
    node_dict: Dict[NodeType, Tensor],
    row_dict: Dict[EdgeType, Tensor],
    col_dict: Dict[EdgeType, Tensor],
    edge_dict: Dict[EdgeType, Tensor],
    perm_dict: Optional[Dict[EdgeType, OptTensor]] = None,
) -> HeteroData:
    # Filters a heterogeneous data object to only hold nodes in `node` and
    # edges in `edge` for each node and edge type, respectively:
    out = copy.copy(data)

    for node_type in out.node_types:
        # Handle the case of disconneted graph sampling:
        if node_type not in node_dict:
            node_dict[node_type] = torch.empty(0, dtype=torch.long)

        filter_node_store_(data[node_type], out[node_type],
                           node_dict[node_type])

    for edge_type in out.edge_types:
        # Handle the case of disconneted graph sampling:
        if edge_type not in row_dict:
            row_dict[edge_type] = torch.empty(0, dtype=torch.long)
        if edge_type not in col_dict:
            col_dict[edge_type] = torch.empty(0, dtype=torch.long)
        if edge_type not in edge_dict:
            edge_dict[edge_type] = torch.empty(0, dtype=torch.long)

        filter_edge_store_(
            data[edge_type],
            out[edge_type],
            row_dict[edge_type],
            col_dict[edge_type],
            edge_dict[edge_type],
            perm_dict.get(edge_type, None) if perm_dict else None,
        )

    return out


def filter_custom_store(
    feature_store: FeatureStore,
    graph_store: GraphStore,
    node_dict: Dict[str, Tensor],
    row_dict: Dict[str, Tensor],
    col_dict: Dict[str, Tensor],
    edge_dict: Dict[str, Tensor],
    custom_cls: Optional[HeteroData] = None,
) -> HeteroData:
    r"""Constructs a `HeteroData` object from a feature store that only holds
    nodes in `node` end edges in `edge` for each node and edge type,
    respectively."""

    # Construct a new `HeteroData` object:
    data = custom_cls() if custom_cls is not None else HeteroData()

    # Filter edge storage:
    # TODO support edge attributes
    for attr in graph_store.get_all_edge_attrs():
        key = attr.edge_type
        if key in row_dict and key in col_dict:
            edge_index = torch.stack([row_dict[key], col_dict[key]], dim=0)
            data[attr.edge_type].edge_index = edge_index

    # Filter node storage:
    required_attrs = []
    for attr in feature_store.get_all_tensor_attrs():
        if attr.group_name in node_dict:
            attr.index = node_dict[attr.group_name]
            required_attrs.append(attr)
            data[attr.group_name].num_nodes = attr.index.size(0)

    # NOTE Here, we utilize `feature_store.multi_get` to give the feature store
    # full control over optimizing how it returns features (since the call is
    # synchronous, this amounts to giving the feature store control over all
    # iteration).
    tensors = feature_store.multi_get_tensor(required_attrs)
    for i, attr in enumerate(required_attrs):
        data[attr.group_name][attr.attr_name] = tensors[i]

    return data


# Input Utilities #############################################################


def get_input_nodes(
    data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
    input_nodes: Union[InputNodes, TensorAttr],
) -> Tuple[Optional[str], Sequence]:
    def to_index(tensor):
        if isinstance(tensor, Tensor) and tensor.dtype == torch.bool:
            return tensor.nonzero(as_tuple=False).view(-1)
        if not isinstance(tensor, Tensor):
            return torch.tensor(tensor, dtype=torch.long)
        return tensor

    if isinstance(data, Data):
        if input_nodes is None:
            return None, torch.arange(data.num_nodes)
        return None, to_index(input_nodes)

    elif isinstance(data, HeteroData):
        assert input_nodes is not None

        if isinstance(input_nodes, str):
            return input_nodes, torch.arange(data[input_nodes].num_nodes)

        assert isinstance(input_nodes, (list, tuple))
        assert len(input_nodes) == 2
        assert isinstance(input_nodes[0], str)

        node_type, input_nodes = input_nodes
        if input_nodes is None:
            return node_type, torch.arange(data[node_type].num_nodes)
        return node_type, to_index(input_nodes)

    else:  # Tuple[FeatureStore, GraphStore]
        feature_store, graph_store = data
        assert input_nodes is not None

        if isinstance(input_nodes, Tensor):
            return None, to_index(input_nodes)

        if isinstance(input_nodes, str):
            return input_nodes, torch.arange(
                remote_backend_utils.num_nodes(feature_store, graph_store,
                                               input_nodes))

        if isinstance(input_nodes, (list, tuple)):
            assert len(input_nodes) == 2
            assert isinstance(input_nodes[0], str)

            node_type, input_nodes = input_nodes
            if input_nodes is None:
                return node_type, torch.arange(
                    remote_backend_utils.num_nodes(feature_store, graph_store,
                                                   input_nodes))
            return node_type, to_index(input_nodes)


def get_edge_label_index(
    data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
    edge_label_index: InputEdges,
) -> Tuple[Optional[str], Tensor]:
    edge_type = None
    if isinstance(data, Data):
        if edge_label_index is None:
            return None, data.edge_index
        return None, edge_label_index

    assert edge_label_index is not None
    assert isinstance(edge_label_index, (list, tuple))

    if isinstance(data, HeteroData):
        if isinstance(edge_label_index[0], str):
            edge_type = edge_label_index
            edge_type = data._to_canonical(*edge_type)
            assert edge_type in data.edge_types
            return edge_type, data[edge_type].edge_index

        assert len(edge_label_index) == 2

        edge_type, edge_label_index = edge_label_index
        edge_type = data._to_canonical(*edge_type)

        if edge_label_index is None:
            return edge_type, data[edge_type].edge_index

        return edge_type, edge_label_index

    else:  # Tuple[FeatureStore, GraphStore]
        _, graph_store = data

        # Need the edge index in COO for LinkNeighborLoader:
        def _get_edge_index(edge_type):
            row_dict, col_dict, _ = graph_store.coo([edge_type])
            row = list(row_dict.values())[0]
            col = list(col_dict.values())[0]
            return torch.stack((row, col), dim=0)

        if isinstance(edge_label_index[0], str):
            edge_type = edge_label_index
            return edge_type, _get_edge_index(edge_type)

        assert len(edge_label_index) == 2
        edge_type, edge_label_index = edge_label_index

        if edge_label_index is None:
            return edge_type, _get_edge_index(edge_type)

        return edge_type, edge_label_index
