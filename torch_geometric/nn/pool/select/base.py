from typing import Optional, Tuple

import torch
from torch import Tensor


class Select(torch.nn.Module):
    r"""An abstract base class implementing custom node selections that map the
    nodes of an input graph to supernodes of the pooled one.

    Specifically, :class:`Select` returns a mapping
    :math:`\mathbf{c} \in {\{ -1, \ldots, C - 1\}}^N`, which assigns each node
    to one of :math:`C` super nodes.
    In addition, :class:`Select` returns the number of super nodes.
    """
    def reset_parameters(self):
        pass

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, int]:
        r"""
        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific graph. (default: :obj:`None`)
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
