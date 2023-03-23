import torch
from torch import Tensor

from torch_geometric.typing import Adj, SparseTensor
from torch_geometric.utils import coalesce, degree
from torch_geometric.utils.to_dense_adj import to_dense_adj


def assortativity(edge_index: Adj) -> float:
    r"""The degree assortativity coefficient from the
    `"Mixing patterns in networks"
    <https://arxiv.org/abs/cond-mat/0209450>`_ paper.
    Assortativity in a network refers to the tendency of nodes to
    connect with other similar nodes over dissimilar nodes.
    It is computed from Pearson correlation coefficient of the node degrees.

    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.

    Returns:
        The value of the degree assortativity coefficient for the input
        graph :math:`\in [-1, 1]`

    Example:

        >>> edge_index = torch.tensor([[0, 1, 2, 3, 2],
        ...                            [1, 2, 0, 1, 3]])
        >>> assortativity(edge_index)
        -0.666667640209198
    """
    if isinstance(edge_index, SparseTensor):
        adj: SparseTensor = edge_index
        row, col, _ = adj.coo()
    else:
        assert isinstance(edge_index, Tensor)
        row, col = edge_index

    device = row.device
    out_deg = degree(row, dtype=torch.long)
    in_deg = degree(col, dtype=torch.long)
    degrees = torch.unique(torch.cat([out_deg, in_deg]))
    mapping = row.new_zeros(degrees.max().item() + 1)
    mapping[degrees] = torch.arange(degrees.size(0), device=device)

    # Compute degree mixing matrix (joint probability distribution) `M`
    num_degrees = degrees.size(0)
    src_deg = mapping[out_deg[row]]
    dst_deg = mapping[in_deg[col]]

    pairs = torch.stack([src_deg, dst_deg], dim=0)
    occurrence = torch.ones(pairs.size(1), device=device)
    pairs, occurrence = coalesce(pairs, occurrence)
    M = to_dense_adj(pairs, edge_attr=occurrence, max_num_nodes=num_degrees)[0]
    # normalization
    M /= M.sum()

    # numeric assortativity coefficient, computed by
    # Pearson correlation coefficient of the node degrees
    x = y = degrees.float()
    a, b = M.sum(0), M.sum(1)

    vara = (a * x**2).sum() - ((a * x).sum())**2
    varb = (b * x**2).sum() - ((b * x).sum())**2
    xy = torch.outer(x, y)
    ab = torch.outer(a, b)
    out = (xy * (M - ab)).sum() / (vara * varb).sqrt()
    return out.item()
