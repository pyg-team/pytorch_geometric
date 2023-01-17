import torch
from torch_scatter import scatter_max, scatter_min

from torch_geometric.typing import Adj, OptTensor, SparseTensor, Tensor


def maximal_independent_set(edge_index: Adj, k: int = 1,
                            rank: OptTensor = None) -> Tensor:
    r"""Returns a Maximal :math:`k`-Independent Set of a graph, i.e., a set of
    nodes (as a tensor mask) such that none of them are :math:`k`-hop
    neighbors, and any node in the graph has a :math:`k`-hop neighbor in the
    returned set.

    If :obj:`rank` is provided, the algorithm greedily selects first the nodes
    with lower corresponding :obj:`rank` value. Otherwise, the nodes are
    selected in their canonical order (i.e., :obj:`rank = torch.arange(n)`).

    This method follows `Blelloch's Alogirithm
    <https://arxiv.org/abs/1202.3205>`_ for :math:`k = 1`, and its
    generalization by `Bacciu et al. <https://arxiv.org/abs/2208.03523>`_ for
    higher values of :math:`k`.

    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        k (int): The :math:`k` value (defaults to 1).
        rank (Tensor): Node rank vector. Must be of size :obj:`(n,)` (defaults
            to :obj:`None`).
    """
    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
        device = edge_index.device()
        n = edge_index.size(0)
    else:
        row, col = edge_index[0], edge_index[1]
        device = row.device
        n = edge_index.max().item() + 1

    if rank is None:
        rank = torch.arange(n, dtype=torch.long, device=device)

    mis = torch.zeros(n, dtype=torch.bool, device=device)
    mask = mis.clone()
    min_rank = rank.clone()

    while not mask.all():
        for _ in range(k):
            min_neigh = torch.full_like(min_rank, fill_value=n)
            scatter_min(min_rank[row], col, out=min_neigh)
            torch.minimum(min_neigh, min_rank, out=min_rank)  # self-loops

        mis = mis | torch.eq(rank, min_rank)
        mask = mis.clone().byte()

        for _ in range(k):
            max_neigh = torch.full_like(mask, fill_value=0)
            scatter_max(mask[row], col, out=max_neigh)
            torch.maximum(max_neigh, mask, out=mask)  # self-loops

        mask = mask.to(dtype=torch.bool)
        min_rank = rank.clone()
        min_rank[mask] = n

    return mis
