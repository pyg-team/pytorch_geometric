from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.typing import PairTensor

from .mask import index_to_mask
from .num_nodes import maybe_num_nodes


def get_num_hops(model: torch.nn.Module) -> int:
    r"""Returns the number of hops the model is aggregating information
    from."""
    from torch_geometric.nn.conv import MessagePassing
    num_hops = 0
    for module in model.modules():
        if isinstance(module, MessagePassing):
            num_hops += 1
    return num_hops


def subgraph(subset: Union[Tensor, List[int]], edge_index: Tensor,
             edge_attr: Optional[Tensor] = None, relabel_nodes: bool = False,
             num_nodes: Optional[int] = None, return_edge_mask: bool = False):
    r"""Returns the induced subgraph of :obj:`(edge_index, edge_attr)`
    containing the nodes in :obj:`subset`.

    Args:
        subset (LongTensor, BoolTensor or [int]): The nodes to keep.
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        return_edge_mask (bool, optional): If set to :obj:`True`, will return
            the edge mask to filter out additional edge features.
            (default: :obj:`False`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """

    device = edge_index.device

    if isinstance(subset, (list, tuple)):
        subset = torch.tensor(subset, dtype=torch.long, device=device)

    if subset.dtype == torch.bool or subset.dtype == torch.uint8:
        num_nodes = subset.size(0)
    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        subset = index_to_mask(subset, size=num_nodes)

    node_mask = subset
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    if relabel_nodes:
        node_idx = torch.zeros(node_mask.size(0), dtype=torch.long,
                               device=device)
        node_idx[subset] = torch.arange(subset.sum().item(), device=device)
        edge_index = node_idx[edge_index]

    if return_edge_mask:
        return edge_index, edge_attr, edge_mask
    else:
        return edge_index, edge_attr


def bipartite_subgraph(subset: Union[PairTensor, Tuple[List[int], List[int]]],
                       edge_index: Tensor, edge_attr: Optional[Tensor] = None,
                       relabel_nodes: bool = False, size: Tuple[int,
                                                                int] = None,
                       return_edge_mask: bool = False):
    r"""Returns the induced subgraph of the bipartite graph
    :obj:`(edge_index, edge_attr)` containing the nodes in :obj:`subset`.

    Args:
        subset (Tuple[Tensor, Tensor] or tuple([int],[int])): The nodes
            to keep.
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        size (tuple, optional): The number of nodes.
            (default: :obj:`None`)
        return_edge_mask (bool, optional): If set to :obj:`True`, will return
            the edge mask to filter out additional edge features.
            (default: :obj:`False`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """

    device = edge_index.device

    if isinstance(subset[0], (list, tuple)):
        subset = (torch.tensor(subset[0], dtype=torch.long, device=device),
                  torch.tensor(subset[1], dtype=torch.long, device=device))

    if subset[0].dtype == torch.bool or subset[0].dtype == torch.uint8:
        size = subset[0].size(0), subset[1].size(0)
    else:
        if size is None:
            size = (maybe_num_nodes(edge_index[0]),
                    maybe_num_nodes(edge_index[1]))
        subset = (index_to_mask(subset[0], size=size[0]),
                  index_to_mask(subset[1], size=size[1]))

    node_mask = subset
    edge_mask = node_mask[0][edge_index[0]] & node_mask[1][edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    if relabel_nodes:
        node_idx_i = torch.zeros(node_mask[0].size(0), dtype=torch.long,
                                 device=device)
        node_idx_j = torch.zeros(node_mask[1].size(0), dtype=torch.long,
                                 device=device)
        node_idx_i[node_mask[0]] = torch.arange(node_mask[0].sum().item(),
                                                device=device)
        node_idx_j[node_mask[1]] = torch.arange(node_mask[1].sum().item(),
                                                device=device)
        edge_index = torch.stack(
            [node_idx_i[edge_index[0]], node_idx_j[edge_index[1]]])

    if return_edge_mask:
        return edge_index, edge_attr, edge_mask
    else:
        return edge_index, edge_attr


def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.

    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask
