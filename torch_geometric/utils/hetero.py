from typing import Dict, List, Set, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.typing import (
    Adj,
    EdgeType,
    NodeType,
    OptTensor,
    SparseTensor,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes_dict


def group_hetero_graph(edge_index_dict, num_nodes_dict=None):
    num_nodes_dict = maybe_num_nodes_dict(edge_index_dict, num_nodes_dict)

    tmp = list(edge_index_dict.values())[0]

    key2int = {}

    cumsum, offset = 0, {}  # Helper data.
    node_types, local_node_indices = [], []
    local2global = {}
    for i, (key, N) in enumerate(num_nodes_dict.items()):
        key2int[key] = i
        node_types.append(tmp.new_full((N, ), i))
        local_node_indices.append(torch.arange(N, device=tmp.device))
        offset[key] = cumsum
        local2global[key] = local_node_indices[-1] + cumsum
        local2global[i] = local2global[key]
        cumsum += N

    node_type = torch.cat(node_types, dim=0)
    local_node_idx = torch.cat(local_node_indices, dim=0)

    edge_indices, edge_types = [], []
    for i, (keys, edge_index) in enumerate(edge_index_dict.items()):
        key2int[keys] = i
        inc = torch.tensor([offset[keys[0]], offset[keys[-1]]]).view(2, 1)
        edge_indices.append(edge_index + inc.to(tmp.device))
        edge_types.append(tmp.new_full((edge_index.size(1), ), i))

    edge_index = torch.cat(edge_indices, dim=-1)
    edge_type = torch.cat(edge_types, dim=0)

    return (edge_index, edge_type, node_type, local_node_idx, local2global,
            key2int)


def get_unused_node_types(node_types: List[NodeType],
                          edge_types: List[EdgeType]) -> Set[NodeType]:
    dst_node_types = set(edge_type[-1] for edge_type in edge_types)
    return set(node_types) - set(dst_node_types)


def check_add_self_loops(module: torch.nn.Module, edge_types: List[EdgeType]):
    is_bipartite = any([key[0] != key[-1] for key in edge_types])
    if is_bipartite and getattr(module, 'add_self_loops', False):
        raise ValueError(
            f"'add_self_loops' attribute set to 'True' on module '{module}' "
            f"for use with edge type(s) '{edge_types}'. This will lead to "
            f"incorrect message passing results.")


def construct_bipartite_edge_index(
    edge_index_dict: Dict[EdgeType, Adj],
    src_offset: Dict[EdgeType, int],
    dst_offset: [NodeType, int],
    edge_attr_dict: Dict[Union[EdgeType, str], Tensor] = None,
) -> Tuple[Adj, OptTensor]:
    """Constructs a tensor of edge indices by concatenating edge indices
    for each edge type. The edge indices are increased by the offset of the
    source and destination nodes."""
    edge_indices: List[Tensor] = []
    if edge_attr_dict is not None:
        edge_attrs: List[Tensor] = []
        use_e_attrs = True
    else:
        use_e_attrs = False

    for edge_type in edge_index_dict.keys():
        _, _, dst_type = edge_type

        edge_index = edge_index_dict[edge_type]

        # (TODO) Add support for SparseTensor w/o converting.
        is_sparse = isinstance(edge_index, SparseTensor)
        if is_sparse:  # Convert to COO
            dst, src, _ = edge_index.coo()
            edge_index = torch.stack([src, dst], dim=0)
        else:
            edge_index = edge_index.clone()
        if use_e_attrs:
            if isinstance(list(edge_attr_dict.keys())[0], tuple):
                edge_attr_i = edge_attr_dict[edge_type].expand(
                    edge_index.size(1), -1)
            else:
                # Param dicts dont like tuple keys
                edge_attr_i = edge_attr_dict['__'.join(edge_type)].expand(
                    edge_index.size(1), -1)

            edge_attrs.append(edge_attr_i)

        # Add offset to edge indices:
        edge_index[0] += src_offset[edge_type]
        edge_index[1] += dst_offset[dst_type]
        edge_indices.append(edge_index)

    # Concatenate all edges and edge tensors:
    if use_e_attrs:
        edge_attr = torch.cat(edge_attrs, dim=0)
    else:
        edge_attr = None
    edge_index = torch.cat(edge_indices, dim=1)

    if is_sparse:
        edge_index = SparseTensor(row=edge_index[1], col=edge_index[0],
                                  value=edge_attr if use_e_attrs else None)

    return edge_index, edge_attr
