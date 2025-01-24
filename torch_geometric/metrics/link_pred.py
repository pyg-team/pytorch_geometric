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


class LinkPredMetric(BaseMetric):
    r"""An abstract class for computing link prediction retrieval metrics.

    Args:
        k (int): The number of top-:math:`k` predictions to evaluate against.
    """
    is_differentiable: bool = False
    full_state_update: bool = False
    higher_is_better: Optional[bool] = None
    weighted: bool

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

    def _update(self, data: LinkPredMetricData) -> None:
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
                (f'{"Weighted" if metric.weighted else ""}'
                 f'{metric.__class__.__name__}@{metric.k}'):
                metric
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

        data = LinkPredMetricData(  # Share metric data across metrics.
            pred_index_mat=pred_index_mat,
            edge_label_index=edge_label_index,
            edge_label_weight=edge_label_weight,
        )

        for metric in self.values():
            if metric.weighted:
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
            if not metric.weighted:
                metric._update(data)
                if WITH_TORCHMETRICS:
                    metric._update_count += 1

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
        pred_rel_mat = data.pred_rel_mat[:, :self.k]
        return pred_rel_mat.sum(dim=-1) / self.k


class LinkPredRecall(LinkPredMetric):
    r"""A link prediction metric to compute Recall @ :math:`k`.

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
    Precision).

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
        self.register_buffer('discount', discount)

        if not weighted:
            self.register_buffer('idcg', cumsum(1.0 / discount))
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
    Reciprocal Rank).

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
