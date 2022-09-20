import copy
from typing import Any, Dict, Optional, Set, Tuple, Union

import torch
from torch import Tensor
from torch_scatter import scatter_min

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.storage import EdgeStorage
from torch_geometric.typing import EdgeType, OptTensor

# Edge Layout Conversion ######################################################


# TODO(manan) deprecate when FeatureStore / GraphStore unification is complete
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
    colptr_dict, row_dict, perm_dict = {}, {}, {}

    for store in data.edge_stores:
        key = store._key
        out = to_csc(store, device, share_memory, is_sorted)
        colptr_dict[key], row_dict[key], perm_dict[key] = out

    return colptr_dict, row_dict, perm_dict


# Edge-based Sampling Utilities ###############################################


def add_negative_samples(
    edge_label_index,
    edge_label,
    edge_label_time,
    num_src_nodes: int,
    num_dst_nodes: int,
    negative_sampling_ratio: float,
):
    """Add negative samples and their `edge_label` and `edge_time`
    if `neg_sampling_ratio > 0`"""
    num_pos_edges = edge_label_index.size(1)
    num_neg_edges = int(num_pos_edges * negative_sampling_ratio)

    if num_neg_edges == 0:
        return edge_label_index, edge_label, edge_label_time

    neg_row = torch.randint(num_src_nodes, (num_neg_edges, ))
    neg_col = torch.randint(num_dst_nodes, (num_neg_edges, ))
    neg_edge_label_index = torch.stack([neg_row, neg_col], dim=0)

    if edge_label_time is not None:
        perm = torch.randperm(num_pos_edges)
        edge_label_time = torch.cat(
            [edge_label_time, edge_label_time[perm[:num_neg_edges]]])

    edge_label_index = torch.cat([
        edge_label_index,
        neg_edge_label_index,
    ], dim=1)

    pos_edge_label = edge_label + 1
    neg_edge_label = edge_label.new_zeros((num_neg_edges, ) +
                                          edge_label.size()[1:])

    edge_label = torch.cat([pos_edge_label, neg_edge_label], dim=0)

    return edge_label_index, edge_label, edge_label_time


def set_node_time_dict(
    node_time_dict,
    input_type: EdgeType,
    edge_label_index,
    edge_label_time,
    num_src_nodes: int,
    num_dst_nodes: int,
):
    """For edges in a batch replace `src` and `dst` node times by the min
    across all edge times."""
    def update_time_(node_time_dict, index, node_type, num_nodes):
        node_time_dict[node_type] = node_time_dict[node_type].clone()
        node_time, _ = scatter_min(edge_label_time, index, dim=0,
                                   dim_size=num_nodes)
        # NOTE We assume that node_time is always less than edge_time.
        index_unique = index.unique()
        node_time_dict[node_type][index_unique] = node_time[index_unique]

    node_time_dict = copy.copy(node_time_dict)
    update_time_(node_time_dict, edge_label_index[0], input_type[0],
                 num_src_nodes)
    update_time_(node_time_dict, edge_label_index[1], input_type[-1],
                 num_dst_nodes)
    return node_time_dict


###############################################################################


def remap_keys(
    original: Dict,
    mapping: Dict,
    exclude: Optional[Set[Any]] = None,
) -> Dict:
    exclude = exclude or set()
    return {(k if k in exclude else mapping[k]): v
            for k, v in original.items()}
