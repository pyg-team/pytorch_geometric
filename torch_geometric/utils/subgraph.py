from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.typing import OptTensor, PairTensor

from .mask import index_to_mask
from .num_nodes import maybe_num_nodes


def get_num_hops(model: torch.nn.Module) -> int:
    r"""Returns the number of hops the model is aggregating information
    from.

    Example:

        >>> class GNN(torch.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.conv1 = GCNConv(3, 16)
        ...         self.conv2 = GCNConv(16, 16)
        ...         self.lin = Linear(16, 2)
        ...
        ...     def forward(self, x, edge_index):
        ...         x = torch.F.relu(self.conv1(x, edge_index))
        ...         x = self.conv2(x, edge_index)
        ...         return self.lin(x)
        >>> get_num_hops(GNN())
        2
    """
    from torch_geometric.nn.conv import MessagePassing
    num_hops = 0
    for module in model.modules():
        if isinstance(module, MessagePassing):
            num_hops += 1
    return num_hops


def subgraph(
    subset: Union[Tensor, List[int]],
    edge_index: Tensor,
    edge_attr: OptTensor = None,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    return_edge_mask: bool = False,
) -> Union[Tuple[Tensor, OptTensor], Tuple[Tensor, OptTensor, OptTensor]]:
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

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        ...                            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5]])
        >>> edge_attr = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        >>> subset = torch.tensor([3, 4, 5])
        >>> subgraph(subset, edge_index, edge_attr)
        (tensor([[3, 4, 4, 5],
                [4, 3, 5, 4]]),
        tensor([ 7.,  8.,  9., 10.]))

        >>> subgraph(subset, edge_index, edge_attr, return_edge_mask=True)
        (tensor([[3, 4, 4, 5],
                [4, 3, 5, 4]]),
        tensor([ 7.,  8.,  9., 10.]),
        tensor([False, False, False, False, False, False,  True,
                True,  True,  True,  False, False]))
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


def bipartite_subgraph(
    subset: Union[PairTensor, Tuple[List[int], List[int]]],
    edge_index: Tensor,
    edge_attr: OptTensor = None,
    relabel_nodes: bool = False,
    size: Optional[Tuple[int, int]] = None,
    return_edge_mask: bool = False,
) -> Union[Tuple[Tensor, OptTensor], Tuple[Tensor, OptTensor, OptTensor]]:
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

    Examples:

        >>> edge_index = torch.tensor([[0, 5, 2, 3, 3, 4, 4, 3, 5, 5, 6],
        ...                            [0, 0, 3, 2, 0, 0, 2, 1, 2, 3, 1]])
        >>> edge_attr = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        >>> subset = (torch.tensor([2, 3, 5]), torch.tensor([2, 3]))
        >>> bipartite_subgraph(subset, edge_index, edge_attr)
        (tensor([[2, 3, 5, 5],
                [3, 2, 2, 3]]),
        tensor([ 3,  4,  9, 10]))

        >>> bipartite_subgraph(subset, edge_index, edge_attr,
        ...                    return_edge_mask=True)
        (tensor([[2, 3, 5, 5],
                [3, 2, 2, 3]]),
        tensor([ 3,  4,  9, 10]),
        tensor([False, False,  True,  True, False, False, False, False,
                True,  True,  False]))
    """

    device = edge_index.device

    src_subset, dst_subset = subset
    if not isinstance(src_subset, Tensor):
        src_subset = torch.tensor(src_subset, dtype=torch.long, device=device)
    if not isinstance(dst_subset, Tensor):
        dst_subset = torch.tensor(dst_subset, dtype=torch.long, device=device)

    if src_subset.dtype != torch.bool:
        src_size = int(edge_index[0].max()) + 1 if size is None else size[0]
        src_subset = index_to_mask(src_subset, size=src_size)
        dst_size = int(edge_index[1].max()) + 1 if size is None else size[1]
        dst_subset = index_to_mask(dst_subset, size=dst_size)

    # node_mask = subset
    edge_mask = src_subset[edge_index[0]] & dst_subset[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    if relabel_nodes:
        node_idx_i = edge_index.new_zeros(src_subset.size(0))
        node_idx_j = edge_index.new_zeros(dst_subset.size(0))
        node_idx_i[src_subset] = torch.arange(int(src_subset.sum()),
                                              device=node_idx_i.device)
        node_idx_j[dst_subset] = torch.arange(int(dst_subset.sum()),
                                              device=node_idx_j.device)
        edge_index = torch.stack([
            node_idx_i[edge_index[0]],
            node_idx_j[edge_index[1]],
        ], dim=0)

    if return_edge_mask:
        return edge_index, edge_attr, edge_mask
    else:
        return edge_index, edge_attr


def k_hop_subgraph(
    node_idx: Union[int, List[int], Tensor],
    num_hops: int,
    edge_index: Tensor,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    flow: str = 'source_to_target',
    directed: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Computes the induced subgraph of :obj:`edge_index` around all nodes in
    :attr:`node_idx` reachable within :math:`k` hops.

    The :attr:`flow` argument denotes the direction of edges for finding
    :math:`k`-hop neighbors. If set to :obj:`"source_to_target"`, then the
    method will find all neighbors that point to the initial set of seed nodes
    in :attr:`node_idx.`
    This mimics the natural flow of message passing in Graph Neural Networks.

    The method returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.

    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central seed
            node(s).
        num_hops (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        directed (bool, optional): If set to :obj:`False`, will include all
            edges between all sampled nodes. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 2, 3, 4, 5],
        ...                            [2, 2, 4, 4, 6, 6]])

        >>> # Center node 6, 2-hops
        >>> subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        ...     6, 2, edge_index, relabel_nodes=True)
        >>> subset
        tensor([2, 3, 4, 5, 6])
        >>> edge_index
        tensor([[0, 1, 2, 3],
                [2, 2, 4, 4]])
        >>> mapping
        tensor([4])
        >>> edge_mask
        tensor([False, False,  True,  True,  True,  True])
        >>> subset[mapping]
        tensor([6])

        >>> edge_index = torch.tensor([[1, 2, 4, 5],
        ...                            [0, 1, 5, 6]])
        >>> (subset, edge_index,
        ...  mapping, edge_mask) = k_hop_subgraph([0, 6], 2,
        ...                                       edge_index,
        ...                                       relabel_nodes=True)
        >>> subset
        tensor([0, 1, 2, 4, 5, 6])
        >>> edge_index
        tensor([[1, 2, 3, 4],
                [0, 1, 4, 5]])
        >>> mapping
        tensor([0, 5])
        >>> edge_mask
        tensor([True, True, True, True])
        >>> subset[mapping]
        tensor([0, 6])
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

    if not directed:
        edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask
