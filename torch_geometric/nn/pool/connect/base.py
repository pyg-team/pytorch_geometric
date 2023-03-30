from typing import Optional, Tuple

import torch
from torch import Tensor


class Connect(torch.nn.Module):
    r"""An abstract base class implementing custom edge connection operators.

    Specifically, :class:`Connect` determines for each pair of supernodes the
    presence or abscene of an edge based on the existing edges between the
    nodes in the two supernodes.
    The operator also computes new coarsened edge features (if present).
    """
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        pass

    def forward(
        self,
        cluster: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Args:
            cluster (torch.Tensor): The mapping from nodes to supernodes.
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
