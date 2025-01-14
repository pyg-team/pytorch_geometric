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
    def k(self) -> int:
        r"""Returns the number of predictions per entity."""
        return self.pred_index_mat.size(1)

    @property
    def pred_rel_mat(self) -> Tensor:
        r"""Returns a matrix indicating the relevance of the `k`-th prediction.
        If :obj:`edge_label_weight` is not given, relevance will be denoted as
        binary.
        """
        if hasattr(self, '_pred_rel_mat'):
            return self._pred_rel_mat  # type: ignore

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
            pred_rel_mat = edge_label_weight[pos].where(pred_rel_mat, 0.0)
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


class LinkPredMetric(BaseMetric):
    r"""An abstract class for computing link prediction retrieval metrics.

    Args:
        k (int): The number of top-:math:`k` predictions to evaluate against.
    """
    is_differentiable: bool = False
    full_state_update: bool = False
    higher_is_better: Optional[bool] = None
    weighted: bool = False

    def __init__(self, k: int) -> None:
        super().__init__()

        if k <= 0:
            raise ValueError(f"'k' needs to be a positive integer in "
                             f"'{self.__class__.__name__}' (got {k})")

        self.k = k

        self.accum: Tensor
        self.total: Tensor

        if WITH_TORCHMETRICS:
            self.add_state('accum', torch.tensor(0.), dist_reduce_fx='sum')
            self.add_state('total', torch.tensor(0), dist_reduce_fx='sum')
        else:
            self.register_buffer('accum', torch.tensor(0.))
            self.register_buffer('total', torch.tensor(0))

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

    def _update(self, data: LinkPredMetricData) -> None
        metric = self._compute(data)

        self.accum += metric.sum()
        self.total += (data.label_count > 0).sum()

    def compute(self) -> Tensor:
        r"""Computes the final metric value."""
        if self.total == 0:
            return torch.zeros_like(self.accum)
        return self.accum / self.total

    def reset(self) -> None:
        r"""Resets metric state variables to their default value."""
        if WITH_TORCHMETRICS:
            super().reset()
        else:
            self.accum.zero_()
            self.total.zero_()

    def _compute(self, data: LinkPredMetricData) -> Tensor:
        r"""Computes the specific metric.
        To be implemented separately for each metric class.

        Args:
            data (LinkPredMetricData): The mini-batch data for computing a link
                prediction metric per example.
        """
        raise NotImplementedError

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
                f'{metric.__class__.__name__}@{metric.k}': metric
                for metric in metrics
            }
        assert len(metrics) > 0
        assert isinstance(metrics, dict)

        for name, metric in metrics.items():
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
        return any([metric.weighted for metric in self.values()])

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

        data = LinkPredMetricData(  # Share metric data for every metric.
            pred_index_mat=pred_index_mat,
            edge_label_index=edge_label_index,
            edge_label_weight=edge_label_weight,
        )

        # pred_rel_mat, y_count = LinkPredMetric._prepare(
        #     pred_index_mat, edge_label_index, edge_label_weight)

        # pred_isin_mat = pred_rel_mat
        # if (pred_isin_mat.dtype != torch.bool
        #         and any([not metric.weighted for metric in self.values()])):
        #     pred_isin_mat = pred_rel_mat != 0.0

        # for metric in self.values():
        #     if metric.weighted:
        #         metric._update_from_prepared(pred_rel_mat, y_count)
        #     else:
        #         metric._update_from_prepared(pred_isin_mat, y_count)

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
    r"""A link prediction metric to compute Precision @ :math:`k`.

    Args:
        k (int): The number of top-:math:`k` predictions to evaluate against.
    """
    higher_is_better: bool = True
    weighted: bool = False

    def _compute(self, data: LinkPredMetricData) -> Tensor:
        return data.pred_rel_mat.sum(dim=-1) / self.k


class LinkPredRecall(LinkPredMetric):
    r"""A link prediction metric to compute Recall @ :math:`k`.

    Args:
        k (int): The number of top-:math:`k` predictions to evaluate against.
    """
    higher_is_better: bool = True
    weighted: bool = False

    def _compute(self, data: LinkPredMetricData) -> Tensor:
        return data.pred_rel_mat.sum(dim=-1) / data.label_count.clamp(min=1e-7)


class LinkPredF1(LinkPredMetric):
    r"""A link prediction metric to compute F1 @ :math:`k`.

    Args:
        k (int): The number of top-:math:`k` predictions to evaluate against.
    """
    higher_is_better: bool = True
    weighted: bool = False

    def _compute(self, data: LinkPredMetricData) -> Tensor:
        isin_count = data.pred_rel_mat.sum(dim=-1)
        precision = isin_count / self.k
        recall = isin_count / data.label_count.clamp(min=1e-7)
        return 2 * precision * recall / (precision + recall).clamp(min=1e-7)


class LinkPredMAP(LinkPredMetric):
    r"""A link prediction metric to compute MAP @ :math:`k` (Mean Average
    Precision).

    Args:
        k (int): The number of top-:math:`k` predictions to evaluate against.
    """
    higher_is_better: bool = True
    weighted: bool = False

    def _compute(self, data: LinkPredMetricData) -> Tensor:
        arange = torch.arange(1, data.k + 1, device=data.pred_index_mat.device)
        cum_precision = data.pred_rel_mat.cumsum(dim=1) / arange
        return ((cum_precision * data.pred_rel_mat).sum(dim=-1) /
                data.label_count.clamp(min=1e-7, max=self.k))


class LinkPredNDCG(LinkPredMetric):
    r"""A link prediction metric to compute the NDCG @ :math:`k` (Normalized
    Discounted Cumulative Gain).

    Args:
        k (int): The number of top-:math:`k` predictions to evaluate against.
        weighted (bool, optional): If set to :obj:`True`, assumes sorted lists
            of ground-truth items according to a relevance score as given by
            :obj:`edge_label_weight`. (default: :obj:`False`)
    """
    higher_is_better: bool = True
    weighted: bool = False

    def __init__(self, k: int, weighted: bool = False):
        super().__init__(k=k)
        self.weighted = weighted

        dtype = torch.get_default_dtype()
        multiplier = 1.0 / torch.arange(2, k + 2, dtype=dtype).log2()

        self.multiplier: Tensor
        self.register_buffer('multiplier', multiplier)

        self.idcg: Tensor
        self.register_buffer('idcg', cumsum(multiplier))

    def _compute(self, data: LinkPredMetricData) -> Tensor:
        multiplier = self.multiplier[:data.k].view(1, -1)
        dcg = (data.pred_rel_mat / denominator).sum(dim=-1)
        idcg = self.idcg[data.label_count.clamp(max=self.k)]

        out = dcg / idcg
        out[out.isnan() | out.isinf()] = 0.0
        return out


class LinkPredMRR(LinkPredMetric):
    r"""A link prediction metric to compute the MRR @ :math:`k` (Mean
    Reciprocal Rank).

    Args:
        k (int): The number of top-:math:`k` predictions to evaluate against.
    """
    higher_is_better: bool = True
    weighted: bool = False

    def _compute(self, data: LinkPredMetricData) -> Tensor:
        arange = torch.arange(1, data.k + 1, device=data.pred_index_mat.device)
        return (data.pred_rel_mat / arange).max(dim=-1)[0]
