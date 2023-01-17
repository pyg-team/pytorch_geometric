import torch
from torch_scatter import scatter_min
from torch_sparse import SparseTensor

from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import maximal_independent_set


def cluster_mis(edge_index: Adj, k: int = 1,
                rank: OptTensor = None) -> PairTensor:
    r"""Computes the Maximal :math:`k`-Independent Set (:math:`k`-MIS)
    clustering of a graph, as defined in `"Generalizing Downsampling from
    Regular Data to Graphs" <https://arxiv.org/abs/2208.03523>`_.

    If :obj:`rank` is provided, the algorithm greedily selects first the nodes
    with lower corresponding :obj:`rank` value. Otherwise, the nodes are
    selected in their canonical order (i.e., :obj:`rank = torch.arange(n)`).

    This method returns both the :math:`k`-MIS and the clustering, where the
    :math:`c`-th cluster refers to the :math:`c`-th element of the
    :math:`k`-MIS.

    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        k (int): The :math:`k` value (defaults to 1).
        rank (Tensor): Node rank vector. Must be of size :obj:`(n,)` (defaults
            to :obj:`None`).

    :rtype: (:class:`ByteTensor`, :class:`LongTensor`)
    """
    mis = maximal_independent_set(edge_index=edge_index, k=k, rank=rank)
    n, device = mis.size(0), mis.device

    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
    else:
        row, col = edge_index[0], edge_index[1]

    if rank is None:
        rank = torch.arange(n, dtype=torch.long, device=device)

    min_rank = torch.full((n, ), fill_value=n, dtype=torch.long, device=device)
    rank_mis = rank[mis]
    min_rank[mis] = rank_mis

    for _ in range(k):
        min_neigh = torch.full_like(min_rank, fill_value=n)
        scatter_min(min_rank[row], col, out=min_neigh)
        torch.minimum(min_neigh, min_rank, out=min_rank)

    _, clusters = torch.unique(min_rank, return_inverse=True)
    perm = torch.argsort(rank_mis)
    return mis, perm[clusters]
