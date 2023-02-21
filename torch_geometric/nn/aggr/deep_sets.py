from typing import Optional

import torch
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation

from ..inits import reset


class DeepSetsAggregation(Aggregation):
    r"""Performs Deep Sets aggregation in which the elements to aggregate are
    first transformed by a Multi-Layer Perceptron (MLP)
    :math:`\phi_{\mathbf{\Theta}}`, summed, and then transformed by another MLP
    :math:`\rho_{\mathbf{\Theta}}`, as suggested in the `"Graph Neural Networks
    with Adaptive Readouts" <https://arxiv.org/abs/2211.04952>`_ paper.

    Args:
        local_nn (torch.nn.Module, optional): The neural network
            :math:`\phi_{\mathbf{\Theta}}`, *e.g.*, defined by
            :class:`torch.nn.Sequential` or
            :class:`torch_geometric.nn.models.MLP`. (default: :obj:`None`)
        global_nn (torch.nn.Module, optional): The neural network
            :math:`\rho_{\mathbf{\Theta}}`, *e.g.*, defined by
            :class:`torch.nn.Sequential` or
            :class:`torch_geometric.nn.models.MLP`. (default: :obj:`None`)
    """
    def __init__(
        self,
        local_nn: Optional[torch.nn.Module] = None,
        global_nn: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.local_nn = local_nn
        self.global_nn = global_nn

    def reset_parameters(self):
        if self.local_nn is not None:
            reset(self.local_nn)
        if self.global_nn is not None:
            reset(self.global_nn)

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        if self.local_nn is not None:
            x = self.local_nn(x)
        x = self.reduce(x, index, ptr, dim_size, dim, reduce='sum')
        if self.global_nn is not None:
            x = self.global_nn(x)
        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(local_nn={self.local_nn}, '
                f'global_nn={self.global_nn})')
