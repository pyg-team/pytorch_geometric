from typing import Optional

import torch
from torch import Tensor

from torch_geometric.nn.aggr import WeightedAggregation
from torch_geometric.utils import softmax


class WeightedQuantileAggregation(WeightedAggregation):
    r"""An aggregation operator that returns the dimension-/feature-wise
    :math:`q`-th quantile of a set :math:`\mathcal{X}` with positive weights
    :math:`\mathbb{w}`. That is, for every feature :math:`d`, it computes

    .. math::
        {\mathrm{Q}_q(\mathcal{X})}_d = \begin{cases}
            x_{\pi_i,d} & i = q \cdot n, \\
            f(x_{\pi_i,d}, x_{\pi_{i+1},d}) & i < q \cdot n < i + 1,\\
        \end{cases}

    where :math:`x_{\pi_1,d} \le \dots \le x_{\pi_i,d} \le \dots \le
    x_{\pi_n,d}` and :math:`f(a, b) = a` is and interpolation
    "lower".

    Args:
        q (float): The quantile value :math:`q`. A scalar in the range
            :math:`[0, 1]`. Implementation is not numerically stable for values
            close to 1.
        fill_value (float, optional): The default value in the case no entry is
            found for a given index (default: :obj:`0.0`).
    """
    def __init__(self, q: float, fill_value: float = 0.0):
        super().__init__()

        if not 0. <= q <= 1.:
            raise ValueError("`q` must be in the range [0, 1].")

        self.q = q
        self.fill_value = fill_value

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                weight: Optional[Tensor] = None, ptr: Optional[Tensor] = None,
                dim_size: Optional[int] = None, dim: int = -2) -> Tensor:

        dim = x.dim() + dim if dim < 0 else dim

        self.assert_index_present(index)
        assert index is not None  # Required for TorchScript.

        if weight is None:
            # TODO: divide by degree
            weight = torch.ones_like(index, dtype=x.dtype)
        elif weight.min() < 0:
            raise ValueError(f"The weights must be positive, but the "
                             f"minimum weight is {weight.min().item()}.")

        # To ensure non-negative edge_weights
        edge_weight = torch.clamp_min(weight, 0)

        count = torch.bincount(index, minlength=dim_size or 0)
        cumsum = torch.cumsum(count, dim=0) - count

        cumweight = torch.zeros(count.shape[0], dtype=edge_weight.dtype,
                                device=x.device)
        cumweight = cumweight.scatter_add(0, index, edge_weight)

        shape = [1] * x.dim()
        shape[dim] = -1

        # We require later the index to be sorted
        index, index_perm = torch.sort(index, dim=dim)
        x = x.take_along_dim(index_perm.view(shape), dim=dim)

        # No gradient required, since only indices are being determined
        with torch.no_grad():
            _, x_perm = torch.sort(x, dim=dim)
            edge_weight = edge_weight[x_perm]
            _, index_perm = torch.sort(index[x_perm], dim=dim, stable=True)

            # Accumulate weights per segment
            edge_weight = edge_weight.take_along_dim(index_perm, dim=0)
            edge_weight = edge_weight / cumweight[index][:, None]
            edge_weight[cumsum[1:]] -= 1
            edge_weight = edge_weight.cumsum(0)

            diff = (edge_weight >= self.q).float()
            diff = diff.diff(dim=dim,
                             prepend=torch.zeros_like(edge_weight[:1]))
            # Either where we accumulate to q or the first element of a row
            first_element_diff = index.diff(
                prepend=torch.zeros_like(index[:1]))
            quantile_mask = ((diff > 0) |
                             ((diff == 0) &
                              (first_element_diff.view(shape) > 0)))
            quantile_index, quantile_dim = torch.where(quantile_mask)

            # Convert back to the original indices
            reverse_perm = x_perm.take_along_dim(index_perm, dim=dim)
            quantile_index = reverse_perm[quantile_index, quantile_dim]

        quantile = x[quantile_index, quantile_dim].reshape((dim_size, -1))

        mask = (count == 0).view(shape)
        quantile = quantile.masked_fill(mask, self.fill_value)
        cumweight = cumweight.view(shape).masked_fill(mask, 1.)

        return cumweight * quantile

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(q={self._q})'


class WeightedMedianAggregation(WeightedQuantileAggregation):
    r"""An aggregation operator that returns the feature-wise weighted median
    of a set. That is, for every feature :math:`d`, it computes

    .. math::
        {\mathrm{median}(\mathcal{X})}_d = x_{\pi_i,d}

    where :math:`x_{\pi_1,d} \le x_{\pi_2,d} \le \dots \le
    x_{\pi_n,d}` and :math:`i = \lfloor \frac{n}{2} \rfloor`.

    .. note::
        If the median lies between two values, the lowest one is returned.
        To compute the midpoint (or other kind of interpolation) of the two
        values, use :class:`QuantileAggregation` instead.

    Args:
        fill_value (float, optional): The default value in the case no entry is
            found for a given index (default: :obj:`0.0`).
    """
    def __init__(self, fill_value: float = 0.0):
        super().__init__(0.5, fill_value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class SoftMedianAggregation(WeightedAggregation):
    r"""The soft median aggregation operator based on a temperature term, as
    described in the `"Robustness of Graph Neural Networks at Scale"
    <https://arxiv.org/abs/2110.14038>`_ paper

    .. math::
        \mathrm{softmax}(\mathcal{X}|t) = \sum_{\mathbf{x}_i\in\mathcal{X}}
        \frac{\exp(t\cdot\mathbf{x}_i)}{\sum_{\mathbf{x}_j\in\mathcal{X}}
        \exp(t\cdot\mathbf{x}_j)}\cdot\mathbf{x}_{i},

    where :math:`t` controls the softness of the softmax when aggregating over
    a set of features :math:`\mathcal{X}`.

    Args:
        T (float, optional): Temperature for SoftMedian aggregation.
            (default: :obj:`1.0`)
        p (float, optional): Norm for distances (via :obj:`torch.norm`).
            (default: :obj:`2.0`)
    """
    def __init__(self, T: float = 1.0, p: float = 2.0):
        super().__init__()

        if not T > 0:
            raise ValueError("Temperature `T` must be > 0.")

        self.T = T
        self.p = p

        self.dimwise_median = WeightedMedianAggregation()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(T={self.T}, p={self.p})"

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                weight: Optional[Tensor] = None, ptr: Optional[Tensor] = None,
                dim_size: Optional[int] = None, dim: int = -2) -> Tensor:
        dim = x.dim() + dim if dim < 0 else dim
        shape = [1] * x.dim()
        shape[dim] = -1

        if weight is None:
            weight = torch.ones_like(index, dtype=x.dtype)

        weight_sums = self.reduce(weight, index, ptr, dim_size, dim,
                                  reduce='sum')

        x_median = self.dimwise_median(x, index, weight, ptr=ptr,
                                       dim_size=dim_size, dim=dim)
        x_median = x_median / weight_sums.view(shape)

        # Distance of each embedding to dimension-wise median
        distances = torch.norm((x_median[index] - x).T, dim=dim, p=self.p)
        distances = distances / pow(x.shape[dim], 1 / self.p)

        # Softmin to find (softly) closest point to dimension-wise median
        soft_weights = softmax(-distances / self.T, index)

        # Final weights to reweigh inputs per segment
        weighted_values = soft_weights * weight
        segment_weight = self.reduce(weighted_values, index, ptr, dim_size,
                                     dim, reduce='sum')
        weights = weighted_values / segment_weight[index] * weight_sums[index]

        return self.weighted_reduce(x, index, weights, ptr, dim_size, dim,
                                    reduce='sum')
