from typing import Optional, Tuple, Union

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
        if pred_index_mat.size(1) != self.k:
            raise ValueError(f"Expected 'pred_index_mat' to hold {self.k} "
                             f"many indices for every entry "
                             f"(got {pred_index_mat.size(1)})")

        # Compute a boolean matrix indicating if the k-th prediction is part of
        # the ground-truth. We do this by flattening both prediction and
        # target indices, and then determining overlaps via `torch.isin`.
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

        # Compute the number of targets per example:
        y_count = scatter(
            torch.ones_like(edge_label_index[0]),
            edge_label_index[0],
            dim=0,
            dim_size=pred_index_mat.size(0),
            reduce='sum',
        )

        metric = self._compute(pred_isin_mat, y_count)

        self.accum += metric.sum()
        self.total += (y_count > 0).sum()

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
        cum_precision = (torch.cumsum(pred_isin_mat, dim=1) /
                         torch.arange(1, self.k + 1, device=y_count.device))
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
        dcg = (pred_isin_mat * self.multiplier.view(1, -1)).sum(dim=-1)
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
        rank = pred_isin_mat.type(torch.uint8).argmax(dim=-1)
        is_correct = pred_isin_mat.gather(1, rank.view(-1, 1)).view(-1)
        reciprocals = 1.0 / (rank + 1)
        reciprocals[~is_correct] = 0.0
        return reciprocals
