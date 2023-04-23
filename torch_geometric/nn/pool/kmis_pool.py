from typing import Callable, Optional, Tuple, Union

import torch
from torch.nn import Module

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.dense import Linear
from torch_geometric.typing import (
    Adj,
    Optional,
    OptTensor,
    PairTensor,
    SparseTensor,
    Tensor,
)
from torch_geometric.utils import maximal_independent_set, scatter

Scorer = Callable[[Tensor, Adj, OptTensor, OptTensor], Tensor]


def maximal_independent_set_cluster(edge_index: Adj, k: int = 1,
                                    num_nodes: Optional[int] = None,
                                    perm: OptTensor = None) -> PairTensor:
    r"""Computes the Maximal :math:`k`-Independent Set (:math:`k`-MIS)
    clustering of a graph, as defined in `"Generalizing Downsampling from
    Regular Data to Graphs" <https://arxiv.org/abs/2208.03523>`_.

    The algorithm greedily selects the nodes in their canonical order. If a
    permutation :obj:`perm` is provided, the nodes are extracted following
    that permutation instead.

    This method returns both the :math:`k`-MIS and the clustering, where the
    :math:`c`-th cluster refers to the :math:`c`-th element of the
    :math:`k`-MIS.

    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        k (int): The :math:`k` value (defaults to 1).
        perm (LongTensor, optional): Permutation vector. Must be of size
            :obj:`(n,)` (defaults to :obj:`None`).

    :rtype: (:class:`ByteTensor`, :class:`LongTensor`)
    """
    mis = maximal_independent_set(edge_index=edge_index, k=k,
                                  num_nodes=num_nodes, perm=perm)
    n, device = mis.size(0), mis.device

    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
    else:
        row, col = edge_index[0], edge_index[1]

    # TODO: Use scatter's `out` and `include_self` arguments,
    #       when available, instead of adding self-loops
    self_loops = torch.arange(n, dtype=torch.long, device=device)
    row, col = torch.cat([row, self_loops]), torch.cat([col, self_loops])

    if perm is None:
        rank = torch.arange(n, dtype=torch.long, device=device)
    else:
        rank = torch.zeros_like(perm)
        rank[perm] = torch.arange(n, dtype=torch.long, device=device)

    min_rank = torch.full((n, ), fill_value=n, dtype=torch.long, device=device)
    rank_mis = rank[mis]
    min_rank[mis] = rank_mis

    for _ in range(k):
        min_rank = scatter(min_rank[row], col, dim_size=n, reduce='min')

    _, clusters = torch.unique(min_rank, return_inverse=True)
    perm = torch.argsort(rank_mis)
    return mis, perm[clusters]


class KMISPooling(Module):
    r"""Maximal :math:`k`-Independent Set (:math:`k`-MIS) pooling operator
    from `"Generalizing Downsampling from Regular Data to Graphs"
    <https://arxiv.org/abs/2208.03523>`_.

    Args:
        in_channels (int, optional): Size of each input sample. Ignored if
            :obj:`scorer` is not :obj:`"linear"`.
        k (int): The :math:`k` value (defaults to 1).
        scorer (str or Callable): A function that computes a score for every
            node. Nodes with higher score will have higher chances to be
            selected by the pooling algorithm. It can be one of the following:

            - :obj:`"linear"` (default): uses a sigmoid-activated linear
              layer to compute the scores

                .. note::
                    :obj:`in_channels` and :obj:`score_passthrough`
                    must be set when using this option.

            - :obj:`"random"`: assigns a score to every node extracted
              uniformly at random from 0 to 1;

            - :obj:`"constant"`: assigns :math:`1` to every node;

            - :obj:`"canonical"`: assigns :math:`-i` to every :math:`i`-th
              node;

            - :obj:`"first"` (or :obj:`"last"`): use the first (resp. last)
              feature dimension of :math:`\mathbf{X}` as node scores;

            - A custom function having as arguments
              :obj:`(x, edge_index, edge_attr, batch)`. It must return a
              one-dimensional :class:`FloatTensor`.

        score_heuristic (str, optional): Apply one of the following heuristic
            to increase the total score of the selected nodes. Given an
            initial score vector :math:`\mathbf{s} \in \mathbb{R}^n`,

            - :obj:`None`: no heuristic is applied;

            - :obj:`"greedy"` (default): compute the updated score
              :math:`\mathbf{s}'` as

                .. math::
                    \mathbf{s}' = \mathbf{s} \oslash (\mathbf{A} +
                    \mathbf{I})^k\mathbf{1},

              where :math:`\oslash` is the element-wise division;

            - :obj:`"w-greedy"`: compute the updated score
              :math:`\mathbf{s}'` as

                .. math::
                    \mathbf{s}' = \mathbf{s} \oslash (\mathbf{A} +
                    \mathbf{I})^k\mathbf{s},

              where :math:`\oslash` is the element-wise division. All scores
              must be strictly positive when using this option.

        score_passthrough (str, optional): Whether to aggregate the node
            scores to the feature vectors, using the function specified by
            :obj:`aggr_score`. If :obj:`"before"`, all the node scores are
            aggregated to their respective feature vector before the cluster
            aggregation. If :obj:`"after"`, the score of the selected nodes are
            aggregated to the feature vectors after the cluster aggregation.
            If :obj:`None`, the score is not aggregated. Defaults to
            :obj:`"before"`.

                .. note::
                    Set this option either to :obj:`"before"` or :obj:`"after"`
                    whenever :obj:`scorer` is :obj:`"linear"` or a
                    :class:`torch.nn.Module`, to make the scoring function
                    end-to-end differentiable.

        aggr_x (str or Aggregation, optional): The aggregation function to be
            applied to the nodes in the same cluster. Can be any string
            admitted by :obj:`scatter` (e.g., :obj:`'sum'`, :obj:`'mean'`,
            :obj:`'max'`) or any :class:`Aggregation`. If :obj:`None`,
            no aggregation is applied and only the nodes in the :math:`k`-MIS
            are selected. Defaults to :obj:`None`.
        aggr_edge (str): The aggregation function to be applied to the edges
            crossing the the same two clusters. Can be any string
            admitted by :obj:`scatter` (e.g., :obj:`'sum'`, :obj:`'mean'`,
            :obj:`'max'`). Defaults to :obj:`'sum'`.
        aggr_score (Callable): The aggregation function to be applied to the
            feature matrix and the score vector. Defaults to :obj:`torch.mul`.
        remove_self_loops (bool): Whether to remove the self-loops from the
            graph after its reduction. Defaults to :obj:`True`.
    """
    _heuristics = {None, 'greedy', 'w-greedy'}
    _passthroughs = {None, 'before', 'after'}
    _scorers = {
        'linear',
        'random',
        'constant',
        'canonical',
        'first',
        'last',
    }

    def __init__(self, in_channels: Optional[int] = None, k: int = 1,
                 scorer: Union[Scorer, str] = 'linear',
                 score_heuristic: Optional[str] = 'greedy',
                 score_passthrough: Optional[str] = 'before',
                 aggr_x: Optional[Union[str, Aggregation]] = None,
                 aggr_edge: str = 'sum',
                 aggr_score: Callable[[Tensor, Tensor], Tensor] = torch.mul,
                 remove_self_loops: bool = True) -> None:
        super(KMISPooling, self).__init__()
        assert score_heuristic in self._heuristics, \
            "Unrecognized `score_heuristic` value."
        assert score_passthrough in self._passthroughs, \
            "Unrecognized `score_passthrough` value."

        if not callable(scorer):
            assert scorer in self._scorers, \
                "Unrecognized `scorer` value."

        self.k = k
        self.scorer = scorer
        self.score_heuristic = score_heuristic
        self.score_passthrough = score_passthrough

        self.aggr_x = aggr_x
        self.aggr_edge = aggr_edge
        self.aggr_score = aggr_score
        self.remove_self_loops = remove_self_loops

        if scorer == 'linear':
            assert self.score_passthrough is not None, \
                "`'score_passthrough'` must not be `None`" \
                " when using `'linear'` scorer"

            self.lin = Linear(in_channels=in_channels, out_channels=1,
                              weight_initializer='uniform')

    def _apply_heuristic(self, x: Tensor, adj: SparseTensor) -> Tensor:
        if self.score_heuristic is None:
            return x

        n = adj.size(0)
        row, col, _ = adj.coo()
        x = x.view(-1)

        if self.score_heuristic == 'greedy':
            k_sums = torch.ones_like(x)
        else:
            k_sums = x.clone()

        for _ in range(self.k):
            k_sums = scatter(k_sums[row], col, dim_size=n, reduce='add')

        return x / k_sums

    def _scorer(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None,
                batch: OptTensor = None) -> Tensor:
        if self.scorer == 'linear':
            return self.lin(x).sigmoid()

        if self.scorer == 'random':
            return torch.rand((x.size(0), 1), device=x.device)

        if self.scorer == 'constant':
            return torch.ones((x.size(0), 1), device=x.device)

        if self.scorer == 'canonical':
            return -torch.arange(x.size(0), device=x.device).view(-1, 1)

        if self.scorer == 'first':
            return x[..., [0]]

        if self.scorer == 'last':
            return x[..., [-1]]

        return self.scorer(x, edge_index, edge_attr, batch)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None,
                batch: OptTensor = None) \
            -> Tuple[Tensor, Adj, OptTensor, OptTensor, Tensor, Tensor]:
        """"""
        adj, n = edge_index, x.size(0)

        if not isinstance(edge_index, SparseTensor):
            adj = SparseTensor.from_edge_index(edge_index, edge_attr, (n, n))

        score = self._scorer(x, edge_index, edge_attr, batch)
        updated_score = self._apply_heuristic(score, adj)
        perm = torch.argsort(updated_score.view(-1), 0, descending=True)

        mis, cluster = maximal_independent_set_cluster(adj, self.k, perm=perm)

        row, col, val = adj.coo()
        c = mis.sum()

        if val is None:
            val = torch.ones_like(row, dtype=torch.float)

        adj = SparseTensor(row=cluster[row], col=cluster[col], value=val,
                           is_sorted=False,
                           sparse_sizes=(c, c)).coalesce(self.aggr_edge)

        if self.remove_self_loops:
            adj = adj.remove_diag()

        if self.score_passthrough == 'before':
            x = self.aggr_score(x, score)

        if self.aggr_x is None:
            x = x[mis]
        elif isinstance(self.aggr_x, str):
            x = scatter(x, cluster, dim=0, dim_size=c, reduce=self.aggr_x)
        else:
            x = self.aggr_x(x, cluster, dim_size=c)

        if self.score_passthrough == 'after':
            x = self.aggr_score(x, score[mis])

        if isinstance(edge_index, SparseTensor):
            edge_index, edge_attr = adj, None
        else:
            row, col, edge_attr = adj.coo()
            edge_index = torch.stack([row, col])

        if batch is not None:
            batch = batch[mis]

        return x, edge_index, edge_attr, batch, mis, cluster

    def __repr__(self):
        if self.scorer == 'linear':
            channels = f"in_channels={self.lin.in_channels}, "
        else:
            channels = ""

        return f'{self.__class__.__name__}({channels}k={self.k})'
