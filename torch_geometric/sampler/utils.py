from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.storage import EdgeStorage
from torch_geometric.typing import NodeType, OptTensor
from torch_geometric.utils import coalesce, index_sort
from torch_geometric.utils.sparse import index2ptr

# Edge Layout Conversion ######################################################


def sort_csc(
    row: Tensor,
    col: Tensor,
    src_node_time: OptTensor = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    if src_node_time is None:
        col, perm = index_sort(col)
        return row[perm], col, perm
    else:
        # We use `np.lexsort` to sort based on multiple keys.
        # TODO There does not seem to exist a PyTorch equivalent yet :(
        perm = np.lexsort([
            src_node_time[row].detach().cpu().numpy(),
            col.detach().cpu().numpy()
        ])
        perm = torch.from_numpy(perm).to(col.device)

        return row[perm], col[perm], perm


# TODO(manan) deprecate when FeatureStore / GraphStore unification is complete
def to_csc(
    data: Union[Data, EdgeStorage],
    device: Optional[torch.device] = None,
    share_memory: bool = False,
    is_sorted: bool = False,
    src_node_time: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, OptTensor]:
    # Convert the graph data into a suitable format for sampling (CSC format).
    # Returns the `colptr` and `row` indices of the graph, as well as an
    # `perm` vector that denotes the permutation of edges.
    # Since no permutation of edges is applied when using `SparseTensor`,
    # `perm` can be of type `None`.
    perm: Optional[Tensor] = None

    if hasattr(data, 'adj'):
        if src_node_time is not None:
            raise NotImplementedError("Temporal sampling via 'SparseTensor' "
                                      "format not yet supported")
        colptr, row, _ = data.adj.csc()

    elif hasattr(data, 'adj_t'):
        if src_node_time is not None:
            # TODO (matthias) This only works when instantiating a
            # `SparseTensor` with `is_sorted=True`. Otherwise, the
            # `SparseTensor` will by default re-sort the neighbors according to
            # column index.
            # As such, we probably want to consider re-adding error:
            # raise NotImplementedError("Temporal sampling via 'SparseTensor' "
            #                           "format not yet supported")
            pass
        colptr, row, _ = data.adj_t.csr()

    elif data.edge_index is not None:
        row, col = data.edge_index
        if not is_sorted:
            row, col, perm = sort_csc(row, col, src_node_time)

        colptr = index2ptr(col, data.size(1))

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
    node_time_dict: Optional[Dict[NodeType, Tensor]] = None,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, OptTensor]]:
    # Convert the heterogeneous graph data into a suitable format for sampling
    # (CSC format).
    # Returns dictionaries holding `colptr` and `row` indices as well as edge
    # permutations for each edge type, respectively.
    colptr_dict, row_dict, perm_dict = {}, {}, {}

    for edge_type, store in data.edge_items():
        src_node_time = (node_time_dict or {}).get(edge_type[0], None)
        out = to_csc(store, device, share_memory, is_sorted, src_node_time)
        colptr_dict[edge_type], row_dict[edge_type], perm_dict[edge_type] = out

    return colptr_dict, row_dict, perm_dict


def to_bidirectional(
    row: Tensor,
    col: Tensor,
    rev_row: Tensor,
    rev_col: Tensor,
    edge_id: OptTensor = None,
    rev_edge_id: OptTensor = None,
) -> Tuple[Tensor, Tensor, OptTensor]:

    assert row.numel() == col.numel()
    assert rev_row.numel() == rev_col.numel()

    edge_index = row.new_empty(2, row.numel() + rev_row.numel())
    edge_index[0, :row.numel()] = row
    edge_index[1, :row.numel()] = col
    edge_index[0, row.numel():] = rev_col
    edge_index[1, row.numel():] = rev_row

    if edge_id is not None:
        edge_id = torch.cat([edge_id, rev_edge_id], dim=0)

    (row, col), edge_id = coalesce(edge_index, edge_id, reduce='any')

    return row, col, edge_id


###############################################################################

X, Y = TypeVar('X'), TypeVar('Y')


def remap_keys(
    inputs: Dict[X, Any],
    mapping: Dict[X, Y],
    exclude: Optional[List[X]] = None,
) -> Dict[Union[X, Y], Any]:
    exclude = exclude or []
    return {(k if k in exclude else mapping.get(k, k)): v
            for k, v in inputs.items()}
