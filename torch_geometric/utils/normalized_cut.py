from typing import Optional

from torch_geometric.utils import degree


def normalized_cut(edge_index, edge_attr, num_nodes: Optional[int] = None):
    r"""Computes the normalized cut :math:`\mathbf{e}_{i,j} \cdot
    \left( \frac{1}{\deg(i)} + \frac{1}{\deg(j)} \right)` of a weighted graph
    given by edge indices and edge attributes.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor): Edge weights or multi-dimensional edge features.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`

    Example:

        >>> edge_index = torch.tensor([[1, 1, 2, 3],
        ...                            [3, 3, 1, 2]])
        >>> edge_attr = torch.tensor([1., 1., 1., 1.])
        >>> normalized_cut(edge_index, edge_attr)
        tensor([1.5000, 1.5000, 2.0000, 1.5000])
    """

    row, col = edge_index[0], edge_index[1]
    deg = 1. / degree(col, num_nodes, edge_attr.dtype)
    deg = deg[row] + deg[col]
    cut = edge_attr * deg
    return cut
