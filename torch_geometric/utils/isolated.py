import torch

from .num_nodes import maybe_num_nodes
from .loop import remove_self_loops


def contains_isolated_nodes(edge_index, num_nodes=None):
    r"""Returns :obj:`True` if the graph given by :attr:`edge_index` contains
    isolated nodes.

    Args:
        edge_index (LongTensor): The edge_indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: bool
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    (row, _), _ = remove_self_loops(edge_index)
    return torch.unique(row).size(0) < num_nodes
