from typing import List, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation


class QuantileAggregation(Aggregation):
    r"""An aggregation operator that returns the feature-wise :math:`q`-th
    quantile of a set :math:`\mathcal{X}`. That is, for every feature
    :math:`d`, it computes

    .. math::
        {\mathrm{Q}_q(\mathcal{X})}_d = \begin{cases}
            x_{\pi_i,d} & i = q \cdot n, \\
            f(x_{\pi_i,d}, x_{\pi_{i+1},d}) & i < q \cdot n < i + 1,\\
        \end{cases}

    where :math:`x_{\pi_1,d} \le \dots \le x_{\pi_i,d} \le \dots \le
    x_{\pi_n,d}` and :math:`f(a, b)` is an interpolation
    function defined by :obj:`interpolation`.

    Args:
        q (float or list): The quantile value(s) :math:`q`. Can be a scalar or
            a list of scalars in the range :math:`[0, 1]`. If more than a
            quantile is passed, the results are concatenated.
        interpolation (str): Interpolation method applied if the quantile point
            :math:`q\cdot n` lies between two values
            :math:`a \le b`. Can be one of the following:

            * :obj:`"lower"`: Returns the one with lowest value.

            * :obj:`"higher"`: Returns the one with highest value.

            * :obj:`"midpoint"`: Returns the average of the two values.

            * :obj:`"nearest"`: Returns the one whose index is nearest to the
              quantile point.

            * :obj:`"linear"`: Returns a linear combination of the two
              elements, defined as
              :math:`f(a, b) = a + (b - a)\cdot(q\cdot n - i)`.

            (default: :obj:`"linear"`)
        fill_value (float, optional): The default value in the case no entry is
            found for a given index (default: :obj:`0.0`).
    """
    interpolations = {'linear', 'lower', 'higher', 'nearest', 'midpoint'}

    def __init__(self, q: Union[float, List[float]],
                 interpolation: str = 'linear', fill_value: float = 0.0):
        super().__init__()

        qs = [q] if not isinstance(q, (list, tuple)) else q
        if len(qs) == 0:
            raise ValueError("Provide at least one quantile value for `q`.")
        if not all(0. <= quantile <= 1. for quantile in qs):
            raise ValueError("`q` must be in the range [0, 1].")
        if interpolation not in self.interpolations:
            raise ValueError(f"Invalid interpolation method "
                             f"got ('{interpolation}')")

        self._q = q
        self.register_buffer('q', torch.Tensor(qs).view(-1, 1))
        self.interpolation = interpolation
        self.fill_value = fill_value

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        dim = x.dim() + dim if dim < 0 else dim

        self.assert_index_present(index)
        assert index is not None  # Required for TorchScript.

        count = torch.bincount(index, minlength=dim_size or 0)
        cumsum = torch.cumsum(count, dim=0) - count

        # In case there exists dangling indices (`dim_size > index.max()`), we
        # need to clamp them to prevent out-of-bound issues:
        if dim_size is not None:
            cumsum = cumsum.clamp(max=x.size(dim) - 1)

        q_point = self.q * (count - 1) + cumsum
        q_point = q_point.t().reshape(-1)

        shape = [1] * x.dim()
        shape[dim] = -1
        index = index.view(shape).expand_as(x)

        # Two sorts: the first one on the value,
        # the second (stable) on the indices:
        x, x_perm = torch.sort(x, dim=dim)
        index = index.take_along_dim(x_perm, dim=dim)
        index, index_perm = torch.sort(index, dim=dim, stable=True)
        x = x.take_along_dim(index_perm, dim=dim)

        # Compute the quantile interpolations:
        if self.interpolation == 'lower':
            quantile = x.index_select(dim, q_point.floor().long())
        elif self.interpolation == 'higher':
            quantile = x.index_select(dim, q_point.ceil().long())
        elif self.interpolation == 'nearest':
            quantile = x.index_select(dim, q_point.round().long())
        else:
            l_quant = x.index_select(dim, q_point.floor().long())
            r_quant = x.index_select(dim, q_point.ceil().long())

            if self.interpolation == 'linear':
                q_frac = q_point.frac().view(shape)
                quantile = l_quant + (r_quant - l_quant) * q_frac
            else:  # 'midpoint'
                quantile = 0.5 * l_quant + 0.5 * r_quant

        # If the number of elements is zero, fill with pre-defined value:
        repeats = self.q.numel()
        mask = (count == 0).repeat_interleave(
            repeats, output_size=repeats * count.numel()).view(shape)
        out = quantile.masked_fill(mask, self.fill_value)

        if self.q.numel() > 1:
            shape = list(out.shape)
            shape = (shape[:dim] + [shape[dim] // self.q.numel(), -1] +
                     shape[dim + 2:])
            out = out.view(shape)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(q={self._q})')


class MedianAggregation(QuantileAggregation):
    r"""An aggregation operator that returns the feature-wise median of a set.
    That is, for every feature :math:`d`, it computes

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
        super().__init__(0.5, 'lower', fill_value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
