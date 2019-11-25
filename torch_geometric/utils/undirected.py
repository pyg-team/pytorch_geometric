import torch
from torch_sparse import coalesce, transpose

from .num_nodes import maybe_num_nodes


def is_undirected(edge_index, edge_attr=None, num_nodes=None):
    r"""Returns :obj:`True` if the graph given by :attr:`edge_index` is
    undirected.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: bool
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes,
                                     num_nodes)

    if edge_attr is None:
        undirected_edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        return edge_index.size(1) == undirected_edge_index.size(1)
    else:
        edge_index_t, edge_attr_t = transpose(edge_index, edge_attr, num_nodes,
                                              num_nodes, coalesced=True)
        index_symmetric = torch.all(edge_index == edge_index_t)
        attr_symmetric = torch.all(edge_attr == edge_attr_t)
        return index_symmetric and attr_symmetric


def to_undirected(edge_index, num_nodes=None):
    r"""Converts the graph given by :attr:`edge_index` to an undirected graph,
    so that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: :class:`LongTensor`
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)

    return edge_index
