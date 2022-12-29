from typing import Optional

import torch
from torch import Tensor
from torch_scatter import scatter


class MeanSubtractionNorm(torch.nn.Module):
    r"""Applies layer normalization by subtracting the mean from the inputs
    as described in the  `"Revisiting 'Over-smoothing' in Deep GCNs"
    <https://arxiv.org/pdf/2003.13663.pdf>`_ paper

    .. math::
        \mathbf{x}_i = \mathbf{x}_i - \frac{1}{|\mathcal{V}|}
        \sum_{j \in \mathcal{V}} \mathbf{x}_j
    """
    def reset_parameters(self):
        pass

    def forward(self, x: Tensor, batch: Optional[Tensor] = None,
                dim_size: Optional[int] = None) -> Tensor:
        """"""
        if batch is None:
            return x - x.mean(dim=0, keepdim=True)

        mean = scatter(x, batch, dim=0, dim_size=dim_size, reduce='mean')
        return x - mean[batch]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
