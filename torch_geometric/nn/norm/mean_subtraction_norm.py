from typing import Optional

import torch
from torch import Tensor

from torch_geometric.utils import scatter


class MeanSubtractionNorm(torch.nn.Module):
    r"""Applies layer normalization by subtracting the mean from the inputs
    as described in the  `"Revisiting 'Over-smoothing' in Deep GCNs"
    <https://arxiv.org/pdf/2003.13663.pdf>`_ paper.

    .. math::
        \mathbf{x}_i = \mathbf{x}_i - \frac{1}{|\mathcal{V}|}
        \sum_{j \in \mathcal{V}} \mathbf{x}_j
    """
    def forward(self, x: Tensor, batch: Optional[Tensor] = None,
                dim_size: Optional[int] = None) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The source tensor.
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example. (default: :obj:`None`)
            dim_size (int, optional): The number of examples :math:`B` in case
                :obj:`batch` is given. (default: :obj:`None`)
        """
        if batch is None:
            return x - x.mean(dim=0, keepdim=True)

        mean = scatter(x, batch, dim=0, dim_size=dim_size, reduce='mean')
        return x - mean[batch]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
