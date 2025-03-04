from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.utils import cumsum, scatter

try:
    import torchmetrics  # noqa
    WITH_TORCHMETRICS = True
    BaseMetric = torchmetrics.Metric
except Exception:
    WITH_TORCHMETRICS = False
    BaseMetric = torch.nn.Module  # type: ignore


@dataclass(repr=False)
class LinkPredMetricData:
    pred_index_mat: Tensor
    edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]]
    edge_label_weight: Optional[Tensor] = None

    @property
    def pred_rel_mat(self) -> Tensor:
        r"""Returns a matrix indicating the relevance of the `k`-th prediction.
        If :obj:`edge_label_weight` is not given, relevance will be denoted as
        binary.
        """
        if hasattr(self, '_pred_rel_mat'):
            return self._pred_rel_mat  # type: ignore

        if self.edge_label_index[1].numel() == 0:
            self._pred_rel_mat = torch.zeros_like(
                self.pred_index_mat,
                dtype=torch.bool if self.edge_label_weight is None else
                torch.get_default_dtype(),
            )
            return self._pred_rel_mat

        # Flatten both prediction and ground-truth indices, and determine
        # overlaps afterwards via `torch.searchsorted`.
        max_index = max(  # type: ignore
            self.pred_index_mat.max()
            if self.pred_index_mat.numel() > 0 else 0,
            self.edge_label_index[1].max()
            if self.edge_label_index[1].numel() > 0 else 0,
        ) + 1
        arange = torch.arange(
            start=0,
            end=max_index * self.pred_index_mat.size(0),  # type: ignore
            step=max_index,  # type: ignore
            device=self.pred_index_mat.device,
        ).view(-1, 1)
        flat_pred_index = (self.pred_index_mat + arange).view(-1)
        flat_label_index = max_index * self.edge_label_index[0]
        flat_label_index = flat_label_index + self.edge_label_index[1]
        flat_label_index, perm = flat_label_index.sort()
        edge_label_weight = self.edge_label_weight
        if edge_label_weight is not None:
            assert edge_label_weight.size() == self.edge_label_index[0].size()
            edge_label_weight = edge_label_weight[perm]

        pos = torch.searchsorted(flat_label_index, flat_pred_index)
        pos = pos.clamp(max=flat_label_index.size(0) - 1)  # Out-of-bounds.

        pred_rel_mat = flat_label_index[pos] == flat_pred_index  # Find matches
        if edge_label_weight is not None:
            pred_rel_mat = edge_label_weight[pos].where(
                pred_rel_mat,
                pred_rel_mat.new_zeros(1),
            )
        pred_rel_mat = pred_rel_mat.view(self.pred_index_mat.size())

        self._pred_rel_mat = pred_rel_mat
        return pred_rel_mat

    @property
    def label_count(self) -> Tensor:
        r"""The number of ground-truth labels for every example."""
        if hasattr(self, '_label_count'):
            return self._label_count  # type: ignore

        label_count = scatter(
            torch.ones_like(self.edge_label_index[0]),
            self.edge_label_index[0],
            dim=0,
            dim_size=self.pred_index_mat.size(0),
            reduce='sum',
        )

        self._label_count = label_count
        return label_count

    @property
    def label_weight_sum(self) -> Tensor:
        r"""The sum of edge label weights for every example."""
        if self.edge_label_weight is None:
            return self.label_count

        if hasattr(self, '_label_weight_sum'):
            return self._label_weight_sum  # type: ignore

        label_weight_sum = scatter(
            self.edge_label_weight,
            self.edge_label_index[0],
            dim=0,
            dim_size=self.pred_index_mat.size(0),
            reduce='sum',
        )

        self._label_weight_sum = label_weight_sum
        return label_weight_sum

    @property
    def edge_label_weight_pos(self) -> Optional[Tensor]:
        r"""Returns the position of edge label weights in descending order
        within example-wise buckets.
        """
        if self.edge_label_weight is None:
            return None

        if hasattr(self, '_edge_label_weight_pos'):
            return self._edge_label_weight_pos  # type: ignore

        # Get the permutation via two sorts: One globally on the weights,
        # followed by a (stable) sort on the example indices.
        perm1 = self.edge_label_weight.argsort(descending=True)
        perm2 = self.edge_label_index[0][perm1].argsort(stable=True)
        perm = perm1[perm2]
        # Invert the permutation to get the final position:
        pos = torch.empty_like(perm)
        pos[perm] = torch.arange(perm.size(0), device=perm.device)
        # Normalize position to zero within all buckets:
        pos = pos - cumsum(self.label_count)[self.edge_label_index[0]]

        self._edge_label_weight_pos = pos
        return pos


class _LinkPredMetric(BaseMetric):
    r"""An abstract class for computing link prediction retrieval metrics.

    Args:
        k (int): The number of top-:math:`k` predictions to evaluate against.
    """
    is_differentiable: bool = False
    full_state_update: bool = False
    higher_is_better: Optional[bool] = None

    def __init__(self, k: int) -> None:
        super().__init__()

        if k <= 0:
            raise ValueError(f"'k' needs to be a positive integer in "
                             f"'{self.__class__.__name__}' (got {k})")

        self.k = k

    def update(
        self,
        pred_index_mat: Tensor,
        edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_label_weight: Optional[Tensor] = None,
    ) -> None:
        r"""Updates the state variables based on the current mini-batch
        prediction.

        :meth:`update` can be repeated multiple times to accumulate the results
        of successive predictions, *e.g.*, inside a mini-batch training or
        evaluation loop.

        Args:
            pred_index_mat (torch.Tensor): The top-:math:`k` predictions of
                every example in the mini-batch with shape
                :obj:`[batch_size, k]`.
            edge_label_index (torch.Tensor): The ground-truth indices for every
                example in the mini-batch, given in COO format of shape
                :obj:`[2, num_ground_truth_indices]`.
            edge_label_weight (torch.Tensor, optional): The weight of the
                ground-truth indices for every example in the mini-batch of
                shape :obj:`[num_ground_truth_indices]`. If given, needs to be
                a vector of positive values. Required for weighted metrics,
                ignored otherwise. (default: :obj:`None`)
        """
        raise NotImplementedError

    def compute(self) -> Tensor:
        r"""Computes the final metric value."""
        raise NotImplementedError

    def reset(self) -> None:
        r"""Resets metric state variables to their default value."""
        if WITH_TORCHMETRICS:
            super().reset()
        else:
            self._reset()

    def _reset(self) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(k={self.k})'


class LinkPredMetric(_LinkPredMetric):
    r"""An abstract class for computing link prediction retrieval metrics.

    Args:
        k (int): The number of top-:math:`k` predictions to evaluate against.
    """
    weighted: bool

    def __init__(self, k: int) -> None:
        super().__init__(k)

        self.accum: Tensor
        self.total: Tensor

        if WITH_TORCHMETRICS:
            self.add_state('accum', torch.tensor(0.), dist_reduce_fx='sum')
            self.add_state('total', torch.tensor(0), dist_reduce_fx='sum')
        else:
            self.register_buffer('accum', torch.tensor(0.), persistent=False)
            self.register_buffer('total', torch.tensor(0), persistent=False)

    def update(
        self,
        pred_index_mat: Tensor,
        edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_label_weight: Optional[Tensor] = None,
    ) -> None:
        if self.weighted and edge_label_weight is None:
            raise ValueError(f"'edge_label_weight' is a required argument for "
                             f"weighted '{self.__class__.__name__}' metrics")
        if not self.weighted:
            edge_label_weight = None

        data = LinkPredMetricData(
            pred_index_mat=pred_index_mat,
            edge_label_index=edge_label_index,
            edge_label_weight=edge_label_weight,
        )
        self._update(data)

    def _update(self, data: LinkPredMetricData) -> None:
        metric = self._compute(data)

        self.accum += metric.sum()
        self.total += (data.label_count > 0).sum()

    def compute(self) -> Tensor:
        if self.total == 0:
            return torch.zeros_like(self.accum)
        return self.accum / self.total

    def _compute(self, data: LinkPredMetricData) -> Tensor:
        r"""Computes the specific metric.
        To be implemented separately for each metric class.

        Args:
            data (LinkPredMetricData): The mini-batch data for computing a link
                prediction metric per example.
        """
        raise NotImplementedError

    def _reset(self) -> None:
        self.accum.zero_()
        self.total.zero_()

    def __repr__(self) -> str:
        weighted_repr = ', weighted=True' if self.weighted else ''
        return f'{self.__class__.__name__}(k={self.k}{weighted_repr})'


class LinkPredMetricCollection(torch.nn.ModuleDict):
    r"""A collection of metrics to reduce and speed-up computation of link
    prediction metrics.

    .. code-block:: python

        from torch_geometric.metrics import (
            LinkPredMAP,
            LinkPredMetricCollection,
            LinkPredPrecision,
            LinkPredRecall,
        )

        metrics = LinkPredMetricCollection([
            LinkPredMAP(k=10),
            LinkPredPrecision(k=100),
            LinkPredRecall(k=50),
        ])

        metrics.update(pred_index_mat, edge_label_index)
        out = metrics.compute()
        metrics.reset()

        print(out)
        >>> {'LinkPredMAP@10': tensor(0.375),
        ...  'LinkPredPrecision@100': tensor(0.127),
        ...  'LinkPredRecall@50': tensor(0.483)}

    Args:
        metrics: The link prediction metrics.
    """
    def __init__(
        self,
        metrics: Union[
            List[LinkPredMetric],
            Dict[str, LinkPredMetric],
        ],
    ) -> None:
        super().__init__()

        if isinstance(metrics, (list, tuple)):
            metrics = {
                (f'{"Weighted" if getattr(metric, "weighted", False) else ""}'
                 f'{metric.__class__.__name__}@{metric.k}'):
                metric
                for metric in metrics
            }
        assert len(metrics) > 0
        assert isinstance(metrics, dict)

        for name, metric in metrics.items():
            assert isinstance(metric, _LinkPredMetric)
            self[name] = metric

    @property
    def max_k(self) -> int:
        r"""The maximum number of top-:math:`k` predictions to evaluate
        against.
        """
        return max([metric.k for metric in self.values()])

    @property
    def weighted(self) -> bool:
        r"""Returns :obj:`True` in case the collection holds at least one
        weighted link prediction metric.
        """
        return any(
            [getattr(metric, 'weighted', False) for metric in self.values()])

    def update(  # type: ignore
        self,
        pred_index_mat: Tensor,
        edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_label_weight: Optional[Tensor] = None,
    ) -> None:
        r"""Updates the state variables based on the current mini-batch
        prediction.

        :meth:`update` can be repeated multiple times to accumulate the results
        of successive predictions, *e.g.*, inside a mini-batch training or
        evaluation loop.

        Args:
            pred_index_mat (torch.Tensor): The top-:math:`k` predictions of
                every example in the mini-batch with shape
                :obj:`[batch_size, k]`.
            edge_label_index (torch.Tensor): The ground-truth indices for every
                example in the mini-batch, given in COO format of shape
                :obj:`[2, num_ground_truth_indices]`.
            edge_label_weight (torch.Tensor, optional): The weight of the
                ground-truth indices for every example in the mini-batch of
                shape :obj:`[num_ground_truth_indices]`. If given, needs to be
                a vector of positive values. Required for weighted metrics,
                ignored otherwise. (default: :obj:`None`)
        """
        if self.weighted and edge_label_weight is None:
            raise ValueError(f"'edge_label_weight' is a required argument for "
                             f"weighted '{self.__class__.__name__}' metrics")
        if not self.weighted:
            edge_label_weight = None

        data = LinkPredMetricData(  # Share metric data across metrics.
            pred_index_mat=pred_index_mat,
            edge_label_index=edge_label_index,
            edge_label_weight=edge_label_weight,
        )

        for metric in self.values():
            if isinstance(metric, LinkPredMetric) and metric.weighted:
                metric._update(data)
                if WITH_TORCHMETRICS:
                    metric._update_count += 1

        data.edge_label_weight = None
        if hasattr(data, '_pred_rel_mat'):
            data._pred_rel_mat = data._pred_rel_mat != 0.0
        if hasattr(data, '_label_weight_sum'):
            del data._label_weight_sum
        if hasattr(data, '_edge_label_weight_pos'):
            del data._edge_label_weight_pos

        for metric in self.values():
            if isinstance(metric, LinkPredMetric) and not metric.weighted:
                metric._update(data)
                if WITH_TORCHMETRICS:
                    metric._update_count += 1

        for metric in self.values():
            if not isinstance(metric, LinkPredMetric):
                metric.update(pred_index_mat, edge_label_index,
                              edge_label_weight)

    def compute(self) -> Dict[str, Tensor]:
        r"""Computes the final metric values."""
        return {name: metric.compute() for name, metric in self.items()}

    def reset(self) -> None:
        r"""Reset metric state variables to their default value."""
        for metric in self.values():
            metric.reset()

    def __repr__(self) -> str:
        names = [f'  {name}: {metric},\n' for name, metric in self.items()]
        return f'{self.__class__.__name__}([\n{"".join(names)}])'


class LinkPredPrecision(LinkPredMetric):
    r"""A link prediction metric to compute Precision @ :math:`k`, *i.e.* the
    proportion of recommendations within the top-:math:`k` that are actually
    relevant.

    A higher precision indicates the model's ability to surface relevant items
    early in the ranking.

    Args:
        k (int): The number of top-:math:`k` predictions to evaluate against.
    """
    higher_is_better: bool = True
    weighted: bool = False

    def _compute(self, data: LinkPredMetricData) -> Tensor:
        pred_rel_mat = data.pred_rel_mat[:, :self.k]
        return pred_rel_mat.sum(dim=-1) / self.k


class LinkPredRecall(LinkPredMetric):
    r"""A link prediction metric to compute Recall @ :math:`k`, *i.e.* the
    proportion of relevant items that appear within the top-:math:`k`.

    A higher recall indicates the model's ability to retrieve a larger
    proportion of relevant items.

    Args:
        k (int): The number of top-:math:`k` predictions to evaluate against.
    """
    higher_is_better: bool = True

    def __init__(self, k: int, weighted: bool = False):
        super().__init__(k=k)
        self.weighted = weighted

    def _compute(self, data: LinkPredMetricData) -> Tensor:
        pred_rel_mat = data.pred_rel_mat[:, :self.k]
        return pred_rel_mat.sum(dim=-1) / data.label_weight_sum.clamp(min=1e-7)


class LinkPredF1(LinkPredMetric):
    r"""A link prediction metric to compute F1 @ :math:`k`.

    Args:
        k (int): The number of top-:math:`k` predictions to evaluate against.
    """
    higher_is_better: bool = True
    weighted: bool = False

    def _compute(self, data: LinkPredMetricData) -> Tensor:
        pred_rel_mat = data.pred_rel_mat[:, :self.k]
        isin_count = pred_rel_mat.sum(dim=-1)
        precision = isin_count / self.k
        recall = isin_count / data.label_count.clamp(min=1e-7)
        return 2 * precision * recall / (precision + recall).clamp(min=1e-7)


class LinkPredMAP(LinkPredMetric):
    r"""A link prediction metric to compute MAP @ :math:`k` (Mean Average
    Precision), considering the order of relevant items within the
    top-:math:`k`.

    MAP @ :math:`k` can provide a more comprehensive view of ranking quality
    than precision alone.

    Args:
        k (int): The number of top-:math:`k` predictions to evaluate against.
    """
    higher_is_better: bool = True
    weighted: bool = False

    def _compute(self, data: LinkPredMetricData) -> Tensor:
        pred_rel_mat = data.pred_rel_mat[:, :self.k]
        device = pred_rel_mat.device
        arange = torch.arange(1, pred_rel_mat.size(1) + 1, device=device)
        cum_precision = pred_rel_mat.cumsum(dim=1) / arange
        return ((cum_precision * pred_rel_mat).sum(dim=-1) /
                data.label_count.clamp(min=1e-7, max=self.k))


class LinkPredNDCG(LinkPredMetric):
    r"""A link prediction metric to compute the NDCG @ :math:`k` (Normalized
    Discounted Cumulative Gain).

    In particular, can account for the position of relevant items by
    considering relevance scores, giving higher weight to more relevant items
    appearing at the top.

    Args:
        k (int): The number of top-:math:`k` predictions to evaluate against.
        weighted (bool, optional): If set to :obj:`True`, assumes sorted lists
            of ground-truth items according to a relevance score as given by
            :obj:`edge_label_weight`. (default: :obj:`False`)
    """
    higher_is_better: bool = True

    def __init__(self, k: int, weighted: bool = False):
        super().__init__(k=k)
        self.weighted = weighted

        dtype = torch.get_default_dtype()
        discount = torch.arange(2, k + 2, dtype=dtype).log2()

        self.discount: Tensor
        self.register_buffer('discount', discount, persistent=False)

        if not weighted:
            self.register_buffer('idcg', cumsum(1.0 / discount),
                                 persistent=False)
        else:
            self.idcg = None

    def _compute(self, data: LinkPredMetricData) -> Tensor:
        pred_rel_mat = data.pred_rel_mat[:, :self.k]
        discount = self.discount[:pred_rel_mat.size(1)].view(1, -1)
        dcg = (pred_rel_mat / discount).sum(dim=-1)

        if not self.weighted:
            assert self.idcg is not None
            idcg = self.idcg[data.label_count.clamp(max=self.k)]
        else:
            assert data.edge_label_weight is not None
            pos = data.edge_label_weight_pos
            assert pos is not None

            discount = torch.cat([
                self.discount,
                self.discount.new_full((1, ), fill_value=float('inf')),
            ])
            discount = discount[pos.clamp(max=self.k)]

            idcg = scatter(  # Apply discount and aggregate:
                data.edge_label_weight / discount,
                data.edge_label_index[0],
                dim_size=data.pred_index_mat.size(0),
                reduce='sum',
            )

        out = dcg / idcg
        out[out.isnan() | out.isinf()] = 0.0
        return out


class LinkPredMRR(LinkPredMetric):
    r"""A link prediction metric to compute the MRR @ :math:`k` (Mean
    Reciprocal Rank), *i.e.* the mean reciprocal rank of the first correct
    prediction (or zero otherwise).

    Args:
        k (int): The number of top-:math:`k` predictions to evaluate against.
    """
    higher_is_better: bool = True
    weighted: bool = False

    def _compute(self, data: LinkPredMetricData) -> Tensor:
        pred_rel_mat = data.pred_rel_mat[:, :self.k]
        device = pred_rel_mat.device
        arange = torch.arange(1, pred_rel_mat.size(1) + 1, device=device)
        return (pred_rel_mat / arange).max(dim=-1)[0]


class LinkPredHitRatio(LinkPredMetric):
    r"""A link prediction metric to compute the hit ratio @ :math:`k`, *i.e.*
    the percentage of users for whom at least one relevant item is present
    within the top-:math:`k` recommendations.

    A high ratio signifies the model's effectiveness in satisfying a broad
    range of user preferences.
    """
    higher_is_better: bool = True
    weighted: bool = False

    def _compute(self, data: LinkPredMetricData) -> Tensor:
        pred_rel_mat = data.pred_rel_mat[:, :self.k]
        return pred_rel_mat.max(dim=-1)[0].to(torch.get_default_dtype())


class LinkPredCoverage(_LinkPredMetric):
    r"""A link prediction metric to compute the Coverage @ :math:`k` of
    predictions, *i.e.* the percentage of unique items recommended across all
    users within the top-:math:`k`.

    Higher coverage indicates a wider exploration of the item catalog.

    Args:
        k (int): The number of top-:math:`k` predictions to evaluate against.
        num_dst_nodes (int): The total number of destination nodes.
    """
    higher_is_better: bool = True

    def __init__(self, k: int, num_dst_nodes: int) -> None:
        super().__init__(k)
        self.num_dst_nodes = num_dst_nodes

        self.mask: Tensor
        mask = torch.zeros(num_dst_nodes, dtype=torch.bool)
        if WITH_TORCHMETRICS:
            self.add_state('mask', mask, dist_reduce_fx='max')
        else:
            self.register_buffer('mask', mask, persistent=False)

    def update(
        self,
        pred_index_mat: Tensor,
        edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_label_weight: Optional[Tensor] = None,
    ) -> None:
        self.mask[pred_index_mat[:, :self.k].flatten()] = True

    def compute(self) -> Tensor:
        return self.mask.to(torch.get_default_dtype()).mean()

    def _reset(self) -> None:
        self.mask.zero_()

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(k={self.k}, '
                f'num_dst_nodes={self.num_dst_nodes})')


class LinkPredDiversity(_LinkPredMetric):
    r"""A link prediction metric to compute the Diversity @ :math:`k` of
    predictions according to item categories.

    Diversity is computed as

    .. math::
        div_{u@k} = 1 - \left( \frac{1}{k \cdot (k-1)} \right) \sum_{i \neq j}
        sim(i, j)

    where

    .. math::
        sim(i,j) = \begin{cases}
            1 & \quad \text{if } i,j \text{ share category,}\\
            0 & \quad \text{otherwise.}
        \end{cases}

    which measures the pair-wise inequality of recommendations according to
    item categories.

    Args:
        k (int): The number of top-:math:`k` predictions to evaluate against.
        category (torch.Tensor): A vector that assigns each destination node to
            a specific category.
    """
    higher_is_better: bool = True

    def __init__(self, k: int, category: Tensor) -> None:
        super().__init__(k)

        if WITH_TORCHMETRICS:
            self.add_state('accum', torch.tensor(0.), dist_reduce_fx='sum')
            self.add_state('total', torch.tensor(0), dist_reduce_fx='sum')
        else:
            self.register_buffer('accum', torch.tensor(0.), persistent=False)
            self.register_buffer('total', torch.tensor(0), persistent=False)

        self.category: Tensor
        self.register_buffer('category', category, persistent=False)

    def update(
        self,
        pred_index_mat: Tensor,
        edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_label_weight: Optional[Tensor] = None,
    ) -> None:
        category = self.category[pred_index_mat[:, :self.k]]

        sim = (category.unsqueeze(-2) == category.unsqueeze(-1)).sum(dim=-1)
        div = 1 - 1 / (self.k * (self.k - 1)) * (sim - 1).sum(dim=-1)

        self.accum += div.sum()
        self.total += pred_index_mat.size(0)

    def compute(self) -> Tensor:
        if self.total == 0:
            return torch.zeros_like(self.accum)
        return self.accum / self.total

    def _reset(self) -> None:
        self.accum.zero_()
        self.total.zero_()


class LinkPredPersonalization(_LinkPredMetric):
    r"""A link prediction metric to compute the Personalization @ :math:`k`,
    *i.e.* the dissimilarity of recommendations across different users.

    Higher personalization suggests that the model tailors recommendations to
    individual user preferences rather than providing generic results.

    Dissimilarity is defined by the average inverse cosine similarity between
    users' lists of recommendations.

    Args:
        k (int): The number of top-:math:`k` predictions to evaluate against.
        max_src_nodes (int, optional): The maximum source nodes to consider to
            compute pair-wise dissimilarity. If specified,
            Personalization @ :math:`k` is approximated to avoid computation
            blowup due to quadratic complexity. (default: :obj:`2**12`)
        batch_size (int, optional): The batch size to determine how many pairs
            of user recommendations should be processed at once.
            (default: :obj:`2**16`)
    """
    higher_is_better: bool = True

    def __init__(
        self,
        k: int,
        max_src_nodes: Optional[int] = 2**12,
        batch_size: int = 2**16,
    ) -> None:
        super().__init__(k)
        self.max_src_nodes = max_src_nodes
        self.batch_size = batch_size

        if WITH_TORCHMETRICS:
            self.add_state('preds', default=[], dist_reduce_fx='cat')
            self.add_state('total', torch.tensor(0), dist_reduce_fx='sum')
        else:
            self.preds: List[Tensor] = []
            self.register_buffer('total', torch.tensor(0), persistent=False)

    def update(
        self,
        pred_index_mat: Tensor,
        edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_label_weight: Optional[Tensor] = None,
    ) -> None:

        # NOTE Move to CPU to avoid memory blowup.
        pred_index_mat = pred_index_mat[:, :self.k].cpu()

        if self.max_src_nodes is None:
            self.preds.append(pred_index_mat)
            self.total += pred_index_mat.size(0)
        elif self.total < self.max_src_nodes:
            remaining = int(self.max_src_nodes - self.total)
            pred_index_mat = pred_index_mat[:remaining]
            self.preds.append(pred_index_mat)
            self.total += pred_index_mat.size(0)

    def compute(self) -> Tensor:
        device = self.total.device
        score = torch.tensor(0.0, device=device)
        total = torch.tensor(0, device=device)

        if len(self.preds) == 0:
            return score

        pred = torch.cat(self.preds, dim=0)

        if pred.size(0) == 0:
            return score

        # Calculate all pairs of nodes (e.g., triu_indices with offset=1).
        # NOTE We do this in chunks to avoid memory blow-up, which leads to a
        # more efficient but trickier implementation.
        num_pairs = (pred.size(0) * (pred.size(0) - 1)) // 2
        offset = torch.arange(pred.size(0) - 1, 0, -1, device=device)
        rowptr = cumsum(offset)
        for start in range(0, num_pairs, self.batch_size):
            end = min(start + self.batch_size, num_pairs)
            idx = torch.arange(start, end, device=device)

            # Find the corresponding row:
            row = torch.searchsorted(rowptr, idx, right=True) - 1
            # Find the corresponding column:
            col = idx - rowptr[row] + (pred.size(0) - offset[row])

            left = pred[row.cpu()].to(device)
            right = pred[col.cpu()].to(device)

            # Use offset to work around applying `isin` along a specific dim:
            i = max(left.max(), right.max()) + 1  # type: ignore
            i = torch.arange(0, i * row.size(0), i, device=device).view(-1, 1)
            isin = torch.isin(left + i, right + i)

            # Compute personalization via average inverse cosine similarity:
            cos = isin.sum(dim=-1) / pred.size(1)
            score += (1 - cos).sum()
            total += cos.numel()

        return score / total

    def _reset(self) -> None:
        self.preds = []
        self.total.zero_()


class LinkPredAveragePopularity(_LinkPredMetric):
    r"""A link prediction metric to compute the Average Recommendation
    Popularity (ARP) @ :math:`k`, which provides insights into the model's
    tendency to recommend popular items by averaging the popularity scores of
    items within the top-:math:`k` recommendations.

    Args:
        k (int): The number of top-:math:`k` predictions to evaluate against.
        popularity (torch.Tensor): The popularity of every item in the training
            set, *e.g.*, the number of times an item has been rated.
    """
    higher_is_better: bool = False

    def __init__(self, k: int, popularity: Tensor) -> None:
        super().__init__(k)

        if WITH_TORCHMETRICS:
            self.add_state('accum', torch.tensor(0.), dist_reduce_fx='sum')
            self.add_state('total', torch.tensor(0), dist_reduce_fx='sum')
        else:
            self.register_buffer('accum', torch.tensor(0.), persistent=False)
            self.register_buffer('total', torch.tensor(0), persistent=False)

        self.popularity: Tensor
        self.register_buffer('popularity', popularity, persistent=False)

    def update(
        self,
        pred_index_mat: Tensor,
        edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]],
        edge_label_weight: Optional[Tensor] = None,
    ) -> None:
        pred_index_mat = pred_index_mat[:, :self.k]
        popularity = self.popularity[pred_index_mat]
        popularity = popularity.to(self.accum.dtype).mean(dim=-1)
        self.accum += popularity.sum()
        self.total += popularity.numel()

    def compute(self) -> Tensor:
        if self.total == 0:
            return torch.zeros_like(self.accum)
        return self.accum / self.total

    def _reset(self) -> None:
        self.accum.zero_()
        self.total.zero_()
