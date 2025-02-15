from typing import Optional

from torch import Tensor

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.utils import degree
from torch_geometric.utils._scatter import broadcast


class VariancePreservingAggregation(Aggregation):
    r"""Performs the Variance Preserving Aggregation (VPA) from the `"GNN-VPA:
    A Variance-Preserving Aggregation Strategy for Graph Neural Networks"
    <https://arxiv.org/abs/2403.04747>`_ paper.

    .. math::
        \mathrm{vpa}(\mathcal{X}) = \frac{1}{\sqrt{|\mathcal{X}|}}
        \sum_{\mathbf{x}_i \in \mathcal{X}} \mathbf{x}_i
    """
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        out = self.reduce(x, index, ptr, dim_size, dim, reduce='sum')

        if ptr is not None:
            count = ptr.diff().to(out.dtype)
        else:
            count = degree(index, dim_size, dtype=out.dtype)

        count = count.sqrt().clamp(min=1.0)
        count = broadcast(count, ref=out, dim=dim)

        return out / count
