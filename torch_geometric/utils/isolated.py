import torch
from torch_geometric.utils import remove_self_loops, segregate_self_loops

from .num_nodes import maybe_num_nodes


def contains_isolated_nodes(edge_index, num_nodes=None):
    r"""Returns :obj:`True` if the graph given by :attr:`edge_index` contains
    isolated nodes.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: bool
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    (row, col), _ = remove_self_loops(edge_index)

    return torch.unique(torch.cat((row, col))).size(0) < num_nodes


def remove_isolated_nodes(edge_index, edge_attr=None, num_nodes=None):
    r"""Removes the isolated nodes from the graph given by :attr:`edge_index`
    with optional edge attributes :attr:`edge_attr`.
    In addition, returns a mask of shape :obj:`[num_nodes]` to manually filter
    out isolated node features later on.
    Self-loops are preserved for non-isolated nodes.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (LongTensor, Tensor, BoolTensor)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    out = segregate_self_loops(edge_index, edge_attr)
    edge_index, edge_attr, loop_edge_index, loop_edge_attr = out

    mask = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
    mask[edge_index.view(-1)] = 1

    assoc = torch.full((num_nodes, ), -1, dtype=torch.long, device=mask.device)
    assoc[mask] = torch.arange(mask.sum(), device=assoc.device)
    edge_index = assoc[edge_index]

    loop_mask = torch.zeros_like(mask)
    loop_mask[loop_edge_index[0]] = 1
    loop_mask = loop_mask & mask
    loop_assoc = torch.full_like(assoc, -1)
    loop_assoc[loop_edge_index[0]] = torch.arange(loop_edge_index.size(1),
                                                  device=loop_assoc.device)
    loop_idx = loop_assoc[loop_mask]
    loop_edge_index = assoc[loop_edge_index[:, loop_idx]]

    edge_index = torch.cat([edge_index, loop_edge_index], dim=1)

    if edge_attr is not None:
        loop_edge_attr = loop_edge_attr[loop_idx]
        edge_attr = torch.cat([edge_attr, loop_edge_attr], dim=0)

    return edge_index, edge_attr, mask
