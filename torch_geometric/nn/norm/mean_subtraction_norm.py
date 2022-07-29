from typing import Optional

import torch
from torch import Tensor
from torch_scatter import scatter


class MeanSubtractionNorm(torch.nn.Module):
    r"""Applies layer normalization by subtracting the mean
    of the inputs, which is useful for reducing over-smoothing.
    Introduced in the "Revisiting 'Over-smoothing' in Deep GCNs"
    <https://arxiv.org/pdf/2003.13663v1.pdf>`_paper.
    """
    def forward(self, x: Tensor, index: Optional[Tensor] = None):
        if index is None:
            x = x - x.mean(dim=0, keepdim=True)
            return x
        else:
            mean = scatter(x, index, dim=0, reduce='mean')
            x = x - mean.index_select(0, index)
            return x
