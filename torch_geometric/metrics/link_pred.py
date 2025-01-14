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


class LinkPredMetric(BaseMetric):
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

        self.accum: Tensor
        self.total: Tensor

        if WITH_TORCHMETRICS:
            self.add_state('accum', torch.tensor(0.), dist_reduce_fx='sum')
            self.add_state('total', torch.tensor(0), dist_reduce_fx='sum')
        else:
            self.register_buffer('accum', torch.tensor(0.))
            self.register_buffer('total', torch.tensor(0))

    @staticmethod
    def _prepare(
        pred_index_mat: Tensor,
        edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]],
    ) -> Tuple[Tensor, Tensor]:
        # Compute a boolean matrix indicating if the `k`-th prediction is part
        # of the ground-truth, as well as the number of ground-truths for every
        # example. We do this by flattening both prediction and ground-truth
        # indices, and then determining overlaps via `torch.isin`.
        max_index = max(  # type: ignore
            pred_index_mat.max() if pred_index_mat.numel() > 0 else 0,
            edge_label_index[1].max()
            if edge_label_index[1].numel() > 0 else 0,
        ) + 1
        arange = torch.arange(
            start=0,
            end=max_index * pred_index_mat.size(0),  # type: ignore
            step=max_index,  # type: ignore
            device=pred_index_mat.device,
        ).view(-1, 1)
        flat_pred_index = (pred_index_mat + arange).view(-1)
        flat_y_index = max_index * edge_label_index[0] + edge_label_index[1]

        pred_isin_mat = torch.isin(flat_pred_index, flat_y_index)
        pred_isin_mat = pred_isin_mat.view(pred_index_mat.size())

        # Compute the number of ground-truths per example:
        y_count = scatter(
            torch.ones_like(edge_label_index[0]),
            edge_label_index[0],
            dim=0,
            dim_size=pred_index_mat.size(0),
            reduce='sum',
        )

        return pred_isin_mat, y_count

    def _update_from_prepared(
        self,
        pred_isin_mat: Tensor,
        y_count: Tensor,
    ) -> None:
        metric = self._compute(pred_isin_mat[:, :self.k], y_count)
        self.accum += metric.sum()
        self.total += (y_count > 0).sum()

    def update(
        self,
        pred_index_mat: Tensor,
        edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]],
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
        """
        pred_isin_mat, y_count = self._prepare(pred_index_mat,
                                               edge_label_index)
        self._update_from_prepared(pred_isin_mat, y_count)

    def compute(self) -> Tensor:
        r"""Computes the final metric value."""
        if self.total == 0:
            return torch.zeros_like(self.accum)
        return self.accum / self.total

    def reset(self) -> None:
        r"""Reset metric state variables to their default value."""
        if WITH_TORCHMETRICS:
            super().reset()
        else:
            self.accum.zero_()
            self.total.zero_()

    def _compute(self, pred_isin_mat: Tensor, y_count: Tensor) -> Tensor:
        r"""Compute the specific metric.
        To be implemented separately for each metric class.

        Args:
            pred_isin_mat (torch.Tensor): A boolean matrix whose :obj:`(i,k)`
                element indicates if the :obj:`k`-th prediction for the
                :obj:`i`-th example is correct or not.
            y_count (torch.Tensor): A vector indicating the number of
                ground-truth labels for each example.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(k={self.k})'


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

    def update(  # type: ignore
        self,
        pred_index_mat: Tensor,
        edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]],
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
        """
        pred_isin_mat, y_count = LinkPredMetric._prepare(
            pred_index_mat, edge_label_index)
        for metric in self.values():
            metric._update_from_prepared(pred_isin_mat, y_count)

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

    def _compute(self, pred_isin_mat: Tensor, y_count: Tensor) -> Tensor:
        return pred_isin_mat.sum(dim=-1) / self.k


class LinkPredRecall(LinkPredMetric):
    r"""A link prediction metric to compute Recall @ :math:`k`.

    Args:
        k (int): The number of top-:math:`k` predictions to evaluate against.
    """
    higher_is_better: bool = True

    def _compute(self, pred_isin_mat: Tensor, y_count: Tensor) -> Tensor:
        return pred_isin_mat.sum(dim=-1) / y_count.clamp(min=1e-7)


class LinkPredF1(LinkPredMetric):
    r"""A link prediction metric to compute F1 @ :math:`k`.

    Args:
        k (int): The number of top-:math:`k` predictions to evaluate against.
    """
    higher_is_better: bool = True

    def _compute(self, pred_isin_mat: Tensor, y_count: Tensor) -> Tensor:
        isin_count = pred_isin_mat.sum(dim=-1)
        precision = isin_count / self.k
        recall = isin_count = isin_count / y_count.clamp(min=1e-7)
        return 2 * precision * recall / (precision + recall).clamp(min=1e-7)


class LinkPredMAP(LinkPredMetric):
    r"""A link prediction metric to compute MAP @ :math:`k` (Mean Average
    Precision).

    Args:
        k (int): The number of top-:math:`k` predictions to evaluate against.
    """
    higher_is_better: bool = True

    def _compute(self, pred_isin_mat: Tensor, y_count: Tensor) -> Tensor:
        device = pred_isin_mat.device
        arange = torch.arange(1, pred_isin_mat.size(1) + 1, device=device)
        cum_precision = pred_isin_mat.cumsum(dim=1) / arange
        return ((cum_precision * pred_isin_mat).sum(dim=-1) /
                y_count.clamp(min=1e-7, max=self.k))


class LinkPredNDCG(LinkPredMetric):
    r"""A link prediction metric to compute the NDCG @ :math:`k` (Normalized
    Discounted Cumulative Gain).

    Args:
        k (int): The number of top-:math:`k` predictions to evaluate against.
    """
    higher_is_better: bool = True

    def __init__(self, k: int):
        super().__init__(k=k)

        dtype = torch.get_default_dtype()
        multiplier = 1.0 / torch.arange(2, k + 2, dtype=dtype).log2()

        self.multiplier: Tensor
        self.register_buffer('multiplier', multiplier)

        self.idcg: Tensor
        self.register_buffer('idcg', cumsum(multiplier))

    def _compute(self, pred_isin_mat: Tensor, y_count: Tensor) -> Tensor:
        multiplier = self.multiplier[:pred_isin_mat.size(1)].view(1, -1)
        dcg = (pred_isin_mat * multiplier).sum(dim=-1)
        idcg = self.idcg[y_count.clamp(max=self.k)]

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

    def _compute(self, pred_isin_mat: Tensor, y_count: Tensor) -> Tensor:
        device = pred_isin_mat.device
        arange = torch.arange(1, pred_isin_mat.size(1) + 1, device=device)
        return (pred_isin_mat / arange).max(dim=-1)[0]
