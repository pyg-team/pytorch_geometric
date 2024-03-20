from typing import Optional

import torch
from torch import Tensor
from torch_geometric.nn.aggr import Aggregation


class VariancePreservingAggregation(Aggregation):

    r"""
    Performs the Variance Preserving Aggregation (VPA) for graph neural networks,
    as described in https://arxiv.org/pdf/2403.04747.pdf

    .. math::
        \mathrm{vpa}(\mathcal{X}) = \frac{1}{\sqrt{|\mathcal{X}|}}
        \sum_{\mathbf{x}_i \in \mathcal{X}} \mathbf{x}_i.
    """

    def forward(
            self,
            x: Tensor,
            index: Optional[Tensor] = None,
            ptr: Optional[Tensor] = None,
            dim_size: Optional[int] = None,
            dim: int = -2
    ) -> Tensor:

        # apply sum aggregation on x
        sum_aggregation = self.reduce(x, index, ptr, dim_size, dim, reduce="sum")

        # count the number of neighbours
        shape = [1 for _ in x.shape]
        shape[dim] = x.shape[dim]
        ones = torch.ones(size=shape, dtype=x.dtype, device=x.device)
        counts = self.reduce(ones, index, ptr, dim_size, dim, reduce="sum")
        # simpler variant that consumes more memory:
        # counts = self.reduce(torch.ones_like(x), index, ptr, dim_size, dim, reduce="sum")

        return torch.nan_to_num(sum_aggregation / torch.sqrt(counts))
