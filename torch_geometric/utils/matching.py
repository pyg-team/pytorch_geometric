from typing import Optional

import torch

from torch_geometric.typing import Adj, OptTensor, SparseTensor, Tensor
from torch_geometric.utils import scatter


def maximal_matching(edge_index: Adj, num_nodes: Optional[int] = None,
                     perm: OptTensor = None) -> Tensor:
    r"""Returns a Maximal Matching of a graph, i.e., a set of edges (as a
    :class:`ByteTensor`) such that none of them are incident to a common
    vertex, and any edge in the graph is incident to an edge in the returned
    set.

    The algorithm greedily selects the edges in their canonical order. If a
    permutation :obj:`perm` is provided, the edges are extracted following
    that permutation instead.

    This method implements `Blelloch's Alogirithm
    <https://arxiv.org/abs/1202.3205>`_.

    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        num_nodes (int, optional): The number of nodes in the graph.
        perm (LongTensor, optional): Permutation vector. Must be of size
            :obj:`(m,)` (defaults to :obj:`None`).

    :rtype: :class:`ByteTensor`
    """
    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
        device = edge_index.device()
        n, m = edge_index.size(0), edge_index.nnz()
    else:
        row, col = edge_index[0], edge_index[1]
        device = row.device
        n, m = num_nodes, row.size(0)

        if n is None:
            n = edge_index.max().item() + 1

    if perm is None:
        rank = torch.arange(m, dtype=torch.long, device=device)
    else:
        rank = torch.zeros_like(perm)
        rank[perm] = torch.arange(m, dtype=torch.long, device=device)

    match = torch.zeros(m, dtype=torch.bool, device=device)
    mask = torch.ones(m, dtype=torch.bool, device=device)

    # TODO: Use scatter's `out` and `include_self` arguments,
    #       when available, instead of adding self-loops
    max_rank = torch.full((n, ), fill_value=n * n, dtype=torch.long,
                          device=device)
    max_idx = torch.arange(n, dtype=torch.long, device=device)

    while mask.any():
        src = torch.cat([rank[mask], rank[mask], max_rank])
        idx = torch.cat([row[mask], col[mask], max_idx])
        node_rank = scatter(src, idx, reduce='min')
        edge_rank = torch.minimum(node_rank[row], node_rank[col])

        match = match | torch.eq(rank, edge_rank)

        unmatched = torch.ones(n, dtype=torch.bool, device=device)
        idx = torch.cat([row[match], col[match]], dim=0)
        unmatched[idx] = False
        mask = mask & unmatched[row] & unmatched[col]

    return match
