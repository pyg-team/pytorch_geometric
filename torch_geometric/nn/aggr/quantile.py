from typing import Optional, Sequence, SupportsFloat, Union

import torch
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation


class QuantileAggregation(Aggregation):
    r"""An aggregation operator that returns the feature-wise :math:`q`-th
    quantile of a set :math:`\mathcal{X}` of :math:`n` elements. That is,
    for every feature :math:`d`, computes

    .. math::
        \big[\mathrm{Q}_q(\mathcal{X})\big]_d = \begin{cases}
            \mathbf{x}_{\pi_i,d} & i = q\cdot n, \\
            f(\mathbf{x}_{\pi_i,d}, \mathbf{x}_{\pi_{i+1},d}) &
                i < q\cdot n < i + 1,\\
        \end{cases}

    where :math:`\mathbf{x}_{\pi_1,d} \le \dots \le \mathbf{x}_{\pi_i,d} \le
    \dots \le \mathbf{x}_{\pi_n,d}` and :math:`f(a, b)` is an interpolation
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

            * :obj:`"linear"` (default): Returns a linear combination of the
              two elments, defined as
              :math:`f(a, b) = a + (b - a)\cdot(q\cdot n - i)`.

        fill_value (float): Default value in the case no entry is
            found for a given index (default: :obj:`NaN`).
    """

    interpolations = {'linear', 'lower', 'higher', 'nearest', 'midpoint'}

    def __init__(self, q: Union[SupportsFloat, Sequence[SupportsFloat]],
                 interpolation: str = 'linear',
                 fill_value: SupportsFloat = float('nan')):
        if isinstance(q, SupportsFloat):
            q = [q]
        if len(q) == 0:
            raise ValueError("Provide at least a quantile value for `q`.")
        if not all(0. <= quantile <= 1. for quantile in q):
            raise ValueError("`q` must be in the range [0, 1].")
        if interpolation not in self.interpolations:
            raise ValueError(f"Invalid '{interpolation}' interpolation method."
                             f" Available interpolations:"
                             f"{self.interpolations}")

        super().__init__()
        self.q = torch.tensor(q, dtype=torch.float,
                              requires_grad=False).view(-1, 1)
        self.interpolation = interpolation
        self.fill_value = fill_value

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        self.assert_index_present(index)
        assert index is not None  # required for TorchScript

        # compute the quantile points
        if dim_size is None:
            dim_size = 0  # default value for `bincount`
        count = torch.bincount(index, minlength=dim_size)
        cumsum = torch.cumsum(count, dim=0) - count
        q_point = self.q * (count - 1) + cumsum  # outer sum/product, QxB
        q_point = q_point.T.reshape(-1)  # BxQ, then flatten

        # shape used for expansions
        # (1, ..., 1, -1, 1, ..., 1)
        shape = [1] * x.dim()
        shape[dim] = -1

        # two sorts: the first one on the value,
        # the second (stable) on the indices
        x, x_perm = torch.sort(x, dim=dim)
        index = index.view(shape).expand_as(x)
        index = index.take_along_dim(x_perm, dim=dim)
        index, index_perm = torch.sort(index, dim=dim, stable=True)
        x = x.take_along_dim(index_perm, dim=dim)

        # compute the quantile interpolations
        if self.interpolation == 'lower':
            idx = torch.floor(q_point).long()
            quantile = x.index_select(dim, idx)
        elif self.interpolation == 'higher':
            idx = torch.ceil(q_point).long()
            quantile = x.index_select(dim, idx)
        elif self.interpolation == 'nearest':
            idx = torch.round(q_point).long()
            quantile = x.index_select(dim, idx)
        else:
            l_idx = torch.floor(q_point).long()
            r_idx = torch.ceil(q_point).long()
            l_quant = x.index_select(dim, l_idx)
            r_quant = x.index_select(dim, r_idx)

            if self.interpolation == 'linear':
                q_frac = torch.frac(q_point).view(shape)
                quantile = l_quant + (r_quant - l_quant) * q_frac
            else:  # 'midpoint'
                quantile = 0.5 * l_quant + 0.5 * r_quant

        # if the number of elements is 0, return 'nan'/fill_value
        num_qs = self.q.size(0)
        out_mask = (count > 0).repeat_interleave(num_qs).view(shape)
        out = torch.where(
            out_mask, quantile,
            torch.tensor([self.fill_value], dtype=quantile.dtype,
                         device=quantile.device))

        if num_qs > 1:  # if > 1, this may generate a new dimension
            out_shape = list(out.shape)
            out_shape = (out_shape[:dim] + [out_shape[dim] // num_qs, -1] +
                         out_shape[dim + 2:])
            return out.view(out_shape)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(q={self.q.view(-1).tolist()}, '
                f'interpolation="{self.interpolation}", '
                f'fill_value={self.fill_value})')


class MedianAggregation(QuantileAggregation):
    r"""An aggregation operator that returns the feature-wise median of a set
    :math:`\mathcal{X}` of :math:`n` elements. That is, for every feature
    :math:`d`, computes

    .. math::
        \big[\mathrm{median}(\mathcal{X})\big]_d = \mathbf{x}_{\pi_i,d}

    where :math:`\mathbf{x}_{\pi_1,d} \le \mathbf{x}_{\pi_2,d} \le \dots \le
    \mathbf{x}_{\pi_n,d}` and :math:`i = \lfloor \frac{n}{2} \rfloor`.

    .. note::
        If the median lies between two values, the lowest one is returned.
        To compute the midpoint (or other kind of interpolation) of the two
        values, use :class:`QuantileAggregation` instead.

    Args:
        fill_value (float, optional): Default value in the case no entry is
            found for a given index (default: :obj:`NaN`).
    """
    def __init__(self, fill_value: float = float('nan')):
        super().__init__(q=0.5, interpolation='lower', fill_value=fill_value)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(fill_value={self.fill_value})'
