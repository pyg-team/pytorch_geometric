from typing import Optional

import torch
from torch import Tensor

from torch_geometric.typing import OptTensor
from torch_geometric.utils import degree


class GraphSizeNorm(torch.nn.Module):
    r"""Applies Graph Size Normalization over each individual graph in a batch
    of node features as described in the
    `"Benchmarking Graph Neural Networks" <https://arxiv.org/abs/2003.00982>`_
    paper.

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x}_i}{\sqrt{|\mathcal{V}|}}
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, batch: OptTensor = None,
                batch_size: Optional[int] = None) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The source tensor.
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example. (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given. (default: :obj:`None`)
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            batch_size = 1

        inv_sqrt_deg = degree(batch, batch_size, dtype=x.dtype).pow(-0.5)
        return x * inv_sqrt_deg.index_select(0, batch).view(-1, 1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
