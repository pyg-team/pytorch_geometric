from typing import Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.nn.conv.utils.inspector import Inspector


class Select(torch.nn.Module):
    r"""An abstract base class implementing custom node selections that map the
    nodes of an input graph to supernodes of the pooled one.

    Specifically, :class:`Select` returns a mapping
    :math:`\mathbf{c} \in {\{ -1, \ldots, C - 1\}}^N`, which assigns each node
    to one of :math:`C` super nodes.
    In addition, :class:`Select` returns the number of super nodes.
    """
    def __init__(self):
        super().__init__()
        self.inspector = Inspector(self)
        self.inspector.inspect(self.forward)
        for pop_arg in ['x', 'edge_index', 'edge_attr', 'batch']:
            self.inspector.params['forward'].pop(pop_arg, None)

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
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
