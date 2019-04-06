import torch
from torch_sparse import coalesce

from .num_nodes import maybe_num_nodes


def dense_to_sparse(tensor):
    r"""Converts a dense adjacency matrix to a sparse adjacency matrix defined
    by edge indices and edge attributes.

    Args:
        tensor (Tensor): The dense adjacency matrix.

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    index = tensor.nonzero().t().contiguous()
    value = tensor[index[0], index[1]]
    index, value = coalesce(index, value, tensor.size(0), tensor.size(1))
    return index, value


def sparse_to_dense(edge_index, edge_attr=None, num_nodes=None):
    r"""Converts a sparse adjacency matrix given by edge indices and edge
    attributes to a dense adjacency matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor): Edge weights or multi-dimensional edge features.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    N = maybe_num_nodes(edge_index, num_nodes)

    if edge_attr is None:
        edge_attr = torch.ones(edge_index.size(1), device=edge_index.device)

    adj = torch.sparse_coo_tensor(edge_index, edge_attr, torch.Size([N, N]))
    return adj.to_dense()
