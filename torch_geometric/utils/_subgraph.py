from typing import List, Literal, Optional, Tuple, Union, overload

import torch
from torch import Tensor

from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.utils import scatter
from torch_geometric.utils.map import map_index
from torch_geometric.utils.mask import index_to_mask
from torch_geometric.utils.num_nodes import maybe_num_nodes


def get_num_hops(model: torch.nn.Module) -> int:
    r"""Returns the number of hops the model is aggregating information
    from.

    .. note::

        This function counts the number of message passing layers as an
        approximation of the total number of hops covered by the model.
        Its output may not necessarily be correct in case message passing
        layers perform multi-hop aggregation, *e.g.*, as in
        :class:`~torch_geometric.nn.conv.ChebConv`.

    Example:
        >>> class GNN(torch.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.conv1 = GCNConv(3, 16)
        ...         self.conv2 = GCNConv(16, 16)
        ...         self.lin = Linear(16, 2)
        ...
        ...     def forward(self, x, edge_index):
        ...         x = self.conv1(x, edge_index).relu()
        ...         x = self.conv2(x, edge_index).relu()
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


@overload
def subgraph(
    subset: Union[Tensor, List[int]],
    edge_index: Tensor,
    edge_attr: OptTensor = ...,
    relabel_nodes: bool = ...,
    num_nodes: Optional[int] = ...,
) -> Tuple[Tensor, OptTensor]:
    pass


@overload
def subgraph(
    subset: Union[Tensor, List[int]],
    edge_index: Tensor,
    edge_attr: OptTensor = ...,
    relabel_nodes: bool = ...,
    num_nodes: Optional[int] = ...,
    *,
    return_edge_mask: Literal[False],
) -> Tuple[Tensor, OptTensor]:
    pass


@overload
def subgraph(
    subset: Union[Tensor, List[int]],
    edge_index: Tensor,
    edge_attr: OptTensor = ...,
    relabel_nodes: bool = ...,
    num_nodes: Optional[int] = ...,
    *,
    return_edge_mask: Literal[True],
) -> Tuple[Tensor, OptTensor, Tensor]:
    pass


def subgraph(
    subset: Union[Tensor, List[int]],
    edge_index: Tensor,
    edge_attr: OptTensor = None,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    *,
    return_edge_mask: bool = False,
) -> Union[Tuple[Tensor, OptTensor], Tuple[Tensor, OptTensor, Tensor]]:
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
            :obj:`max(edge_index) + 1`. (default: :obj:`None`)
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

    if subset.dtype != torch.bool:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        node_mask = index_to_mask(subset, size=num_nodes)
    else:
        num_nodes = subset.size(0)
        node_mask = subset
        subset = node_mask.nonzero().view(-1)

    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    if relabel_nodes:
        edge_index, _ = map_index(
            edge_index.view(-1),
            subset,
            max_index=num_nodes,
            inclusive=True,
        )
        edge_index = edge_index.view(2, -1)

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
        src_node_mask = index_to_mask(src_subset, size=src_size)
    else:
        src_size = src_subset.size(0)
        src_node_mask = src_subset
        src_subset = src_subset.nonzero().view(-1)

    if dst_subset.dtype != torch.bool:
        dst_size = int(edge_index[1].max()) + 1 if size is None else size[1]
        dst_node_mask = index_to_mask(dst_subset, size=dst_size)
    else:
        dst_size = dst_subset.size(0)
        dst_node_mask = dst_subset
        dst_subset = dst_subset.nonzero().view(-1)

    edge_mask = src_node_mask[edge_index[0]] & dst_node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    if relabel_nodes:
        src_index, _ = map_index(edge_index[0], src_subset, max_index=src_size,
                                 inclusive=True)
        dst_index, _ = map_index(edge_index[1], dst_subset, max_index=dst_size,
                                 inclusive=True)
        edge_index = torch.stack([src_index, dst_index], dim=0)

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
        flow (str, optional): The flow direction of :math:`k`-hop aggregation
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            (default: :obj:`"source_to_target"`)
        directed (bool, optional): If set to :obj:`True`, will only include
            directed edges to the seed nodes :obj:`node_idx`.
            (default: :obj:`False`)

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

    if isinstance(node_idx, int):
        node_idx = torch.tensor([node_idx], device=row.device)
    elif isinstance(node_idx, (list, tuple)):
        node_idx = torch.tensor(node_idx, device=row.device)
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    preserved_edge_mask = torch.zeros_like(edge_mask)
    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        preserved_edge_mask |= edge_mask
        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True

    if not directed:
        edge_mask = node_mask[row] & node_mask[col]
    else:
        edge_mask = preserved_edge_mask

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        mapping = row.new_full((num_nodes, ), -1)
        mapping[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = mapping[edge_index]

    return subset, edge_index, inv, edge_mask


@overload
def hyper_subgraph(
    subset: Union[Tensor, List[int]],
    edge_index: Tensor,
    edge_attr: OptTensor = ...,
    relabel_nodes: bool = ...,
    num_nodes: Optional[int] = ...,
) -> Tuple[Tensor, OptTensor]:
    pass


@overload
def hyper_subgraph(
    subset: Union[Tensor, List[int]],
    edge_index: Tensor,
    edge_attr: OptTensor = ...,
    relabel_nodes: bool = ...,
    num_nodes: Optional[int] = ...,
    *,
    return_edge_mask: Literal[False],
) -> Tuple[Tensor, OptTensor]:
    pass


@overload
def hyper_subgraph(
    subset: Union[Tensor, List[int]],
    edge_index: Tensor,
    edge_attr: OptTensor = ...,
    relabel_nodes: bool = ...,
    num_nodes: Optional[int] = ...,
    *,
    return_edge_mask: Literal[True],
) -> Tuple[Tensor, OptTensor, Tensor]:
    pass


def hyper_subgraph(
    subset: Union[Tensor, List[int]],
    edge_index: Tensor,
    edge_attr: OptTensor = None,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    return_edge_mask: bool = False,
) -> Union[Tuple[Tensor, OptTensor], Tuple[Tensor, OptTensor, Tensor]]:
    r"""Returns the induced subgraph of the hyper graph of
    :obj:`(edge_index, edge_attr)` containing the nodes in :obj:`subset`.

    Args:
        subset (torch.Tensor or [int]): The nodes to keep.
        edge_index (LongTensor): Hyperedge tensor
            with shape :obj:`[2, num_edges*num_nodes_per_edge]`, where
            :obj:`edge_index[1]` denotes the hyperedge index and
            :obj:`edge_index[0]` denotes the node indices that are connected
            by the hyperedge.
        edge_attr (torch.Tensor, optional): Edge weights or multi-dimensional
            edge features of shape :obj:`[num_edges, *]`.
            (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the
            resulting :obj:`edge_index` will be relabeled to hold
            consecutive indices starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max(edge_index[0]) + 1`. (default: :obj:`None`)
        return_edge_mask (bool, optional): If set to :obj:`True`, will return
            the edge mask to filter out additional edge features.
            (default: :obj:`False`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)

    Examples:
        >>> edge_index = torch.tensor([[0, 1, 2, 1, 2, 3, 0, 2, 3],
        ...                            [0, 0, 0, 1, 1, 1, 2, 2, 2]])
        >>> edge_attr = torch.tensor([3, 2, 6])
        >>> subset = torch.tensor([0, 3])
        >>> subgraph(subset, edge_index, edge_attr)
        (tensor([[0, 3],
                [0, 0]]),
        tensor([ 6.]))

        >>> subgraph(subset, edge_index, edge_attr, return_edge_mask=True)
        (tensor([[0, 3],
                [0, 0]]),
        tensor([ 6.]))
        tensor([False, False, True])
    """
    device = edge_index.device

    if isinstance(subset, (list, tuple)):
        subset = torch.tensor(subset, dtype=torch.long, device=device)

    if subset.dtype != torch.bool:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        node_mask = index_to_mask(subset, size=num_nodes)
    else:
        num_nodes = subset.size(0)
        node_mask = subset

    # Mask all connections that contain a node not in the subset
    hyper_edge_connection_mask = node_mask[
        edge_index[0]]  # num_edges*num_nodes_per_edge

    # Mask hyperedges that contain one or less nodes from the subset
    edge_mask = scatter(hyper_edge_connection_mask.to(torch.long),
                        edge_index[1], reduce='sum') > 1

    # Mask connections if hyperedge contains one or less nodes from the subset
    # or is connected to a node not in the subset
    hyper_edge_connection_mask = hyper_edge_connection_mask & edge_mask[
        edge_index[1]]

    edge_index = edge_index[:, hyper_edge_connection_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    # Relabel edges
    edge_idx = torch.zeros(edge_mask.size(0), dtype=torch.long, device=device)
    edge_idx[edge_mask] = torch.arange(edge_mask.sum().item(), device=device)
    edge_index = torch.cat(
        [edge_index[0].unsqueeze(0), edge_idx[edge_index[1]].unsqueeze(0)], 0)

    if relabel_nodes:
        node_idx = torch.zeros(node_mask.size(0), dtype=torch.long,
                               device=device)
        node_idx[subset] = torch.arange(node_mask.sum().item(), device=device)
        edge_index = torch.cat(
            [node_idx[edge_index[0]].unsqueeze(0), edge_index[1].unsqueeze(0)],
            0)

    if return_edge_mask:
        return edge_index, edge_attr, edge_mask
    else:
        return edge_index, edge_attr
