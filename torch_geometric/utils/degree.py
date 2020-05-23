import torch

from .num_nodes import maybe_num_nodes
from typing import Optional


def degree(index,
           num_nodes: Optional[int] = None,
           dtype: Optional[int] = None):
    r"""Computes the (unweighted) degree of a given one-dimensional index
    tensor.

    Args:
        index (LongTensor): Index tensor.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dtype (:obj:`torch.dtype`, optional): The desired data type of the
            returned tensor.

    :rtype: :class:`Tensor`
    """
    num_nodes_int = maybe_num_nodes(index, num_nodes)
    out = torch.zeros((num_nodes_int), dtype=dtype, device=index.device)
    out_ones = torch.ones((index.size(0)), dtype=out.dtype, device=out.device)
    return out.scatter_add_(0, index, out_ones)
