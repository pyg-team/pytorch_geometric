from typing import Callable, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import coalesce, maximal_matching, scatter, softmax


def maximal_matching_cluster(edge_index: Adj, num_nodes: Optional[int] = None,
                             perm: OptTensor = None) -> Tuple[Tensor, Tensor]:
    r"""Computes the Maximal Matching clustering of a graph, where the
    matched edges form 2-element clusters while unmatched vertices are treated
    as singletons.

    The algorithm greedily selects the edges in their canonical order. If a
    permutation :obj:`perm` is provided, the nodes are extracted following
    that permutation instead.

    This method returns both the matching and the clustering.

    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        num_nodes (int, optional): The number of nodes in the graph.
        perm (LongTensor, optional): Permutation vector. Must be of size
            :obj:`(m,)` (defaults to :obj:`None`).

    :rtype: (:class:`ByteTensor`, :class:`LongTensor`)
    """
    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
        device = edge_index.device()
        n = edge_index.size(0)
    else:
        row, col = edge_index[0], edge_index[1]
        device = row.device
        n = num_nodes

        if n is None:
            n = edge_index.max().item() + 1

    match = maximal_matching(edge_index, num_nodes, perm)
    cluster = torch.arange(n, dtype=torch.long, device=device)
    cluster[col[match]] = row[match]

    _, cluster = torch.unique(cluster, return_inverse=True)
    return match, cluster


class UnpoolInfo(NamedTuple):
    edge_index: Tensor
    cluster: Tensor
    batch: Tensor
    new_edge_score: Tensor


class EdgePooling(torch.nn.Module):
    r"""The edge pooling operator from the `"Towards Graph Pooling by Edge
    Contraction" <https://graphreason.github.io/papers/17.pdf>`__ and
    `"Edge Contraction Pooling for Graph Neural Networks"
    <https://arxiv.org/abs/1905.10990>`__ papers.

    In short, a score is computed for each edge.
    Edges are contracted iteratively according to that score unless one of
    their nodes has already been part of a contracted edge.

    To duplicate the configuration from the `"Towards Graph Pooling by Edge
    Contraction" <https://graphreason.github.io/papers/17.pdf>`__ paper, use
    either :func:`EdgePooling.compute_edge_score_softmax`
    or :func:`EdgePooling.compute_edge_score_tanh`, and set
    :obj:`add_to_edge_score` to :obj:`0.0`.

    To duplicate the configuration from the `"Edge Contraction Pooling for
    Graph Neural Networks" <https://arxiv.org/abs/1905.10990>`__ paper,
    set :obj:`dropout` to :obj:`0.2`.

    Args:
        in_channels (int): Size of each input sample.
        edge_score_method (callable, optional): The function to apply
            to compute the edge score from raw edge scores. By default,
            this is the softmax over all incoming edges for each node.
            This function takes in a :obj:`raw_edge_score` tensor of shape
            :obj:`[num_nodes]`, an :obj:`edge_index` tensor and the number of
            nodes :obj:`num_nodes`, and produces a new tensor of the same size
            as :obj:`raw_edge_score` describing normalized edge scores.
            Included functions are
            :func:`EdgePooling.compute_edge_score_softmax`,
            :func:`EdgePooling.compute_edge_score_tanh`, and
            :func:`EdgePooling.compute_edge_score_sigmoid`.
            (default: :func:`EdgePooling.compute_edge_score_softmax`)
        dropout (float, optional): The probability with
            which to drop edge scores during training. (default: :obj:`0.0`)
        add_to_edge_score (float, optional): A value to be added to each
            computed edge score. Adding this greatly helps with unpooling
            stability. (default: :obj:`0.5`)
    """
    def __init__(
        self,
        in_channels: int,
        edge_score_method: Optional[Callable] = None,
        dropout: Optional[float] = 0.0,
        add_to_edge_score: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        if edge_score_method is None:
            edge_score_method = self.compute_edge_score_softmax
        self.compute_edge_score = edge_score_method
        self.add_to_edge_score = add_to_edge_score
        self.dropout = dropout

        self.lin = torch.nn.Linear(2 * in_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin.reset_parameters()

    @staticmethod
    def compute_edge_score_softmax(
        raw_edge_score: Tensor,
        edge_index: Tensor,
        num_nodes: int,
    ) -> Tensor:
        r"""Normalizes edge scores via softmax application."""
        return softmax(raw_edge_score, edge_index[1], num_nodes=num_nodes)

    @staticmethod
    def compute_edge_score_tanh(
        raw_edge_score: Tensor,
        edge_index: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        r"""Normalizes edge scores via hyperbolic tangent application."""
        return torch.tanh(raw_edge_score)

    @staticmethod
    def compute_edge_score_sigmoid(
        raw_edge_score: Tensor,
        edge_index: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        r"""Normalizes edge scores via sigmoid application."""
        return torch.sigmoid(raw_edge_score)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, UnpoolInfo]:
        r"""
        Args:
            x (torch.Tensor): The node features.
            edge_index (torch.Tensor): The edge indices.
            batch (torch.Tensor): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example.

        Return types:
            * **x** *(torch.Tensor)* - The pooled node features.
            * **edge_index** *(torch.Tensor)* - The coarsened edge indices.
            * **batch** *(torch.Tensor)* - The coarsened batch vector.
            * **unpool_info** *(UnpoolInfo)* - Information that is
              consumed by :func:`EdgePooling.unpool` for unpooling.
        """
        e = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        e = self.lin(e).view(-1)
        e = F.dropout(e, p=self.dropout, training=self.training)
        e = self.compute_edge_score(e, edge_index, x.size(0))
        e = e + self.add_to_edge_score

        x, edge_index, batch, unpool_info = self._merge_edges(
            x, edge_index, batch, e)

        return x, edge_index, batch, unpool_info

    def _merge_edges(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_score: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, UnpoolInfo]:
        perm = torch.argsort(edge_score, descending=True)
        match, cluster = maximal_matching_cluster(edge_index,
                                                  num_nodes=x.size(0),
                                                  perm=perm)
        c = cluster.max() + 1

        # We compute the new features as an addition of the old ones.
        new_x = scatter(x, cluster, dim=0, dim_size=c, reduce='sum')
        new_edge_score = torch.ones(c, dtype=x.dtype, device=x.device)

        new_edge_score[cluster[edge_index[0, match]]] = edge_score[match]
        new_x = new_x * new_edge_score.view(-1, 1)

        new_edge_index = coalesce(cluster[edge_index], num_nodes=c)
        new_batch = x.new_empty(c, dtype=torch.long)
        new_batch = new_batch.scatter_(0, cluster, batch)

        unpool_info = UnpoolInfo(edge_index=edge_index, cluster=cluster,
                                 batch=batch, new_edge_score=new_edge_score)

        return new_x, new_edge_index, new_batch, unpool_info

    def unpool(
        self,
        x: Tensor,
        unpool_info: UnpoolInfo,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Unpools a previous edge pooling step.

        For unpooling, :obj:`x` should be of same shape as those produced by
        this layer's :func:`forward` function. Then, it will produce an
        unpooled :obj:`x` in addition to :obj:`edge_index` and :obj:`batch`.

        Args:
            x (torch.Tensor): The node features.
            unpool_info (UnpoolInfo): Information that has been produced by
                :func:`EdgePooling.forward`.

        Return types:
            * **x** *(torch.Tensor)* - The unpooled node features.
            * **edge_index** *(torch.Tensor)* - The new edge indices.
            * **batch** *(torch.Tensor)* - The new batch vector.
        """
        new_x = x / unpool_info.new_edge_score.view(-1, 1)
        new_x = new_x[unpool_info.cluster]
        return new_x, unpool_info.edge_index, unpool_info.batch

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels})'
