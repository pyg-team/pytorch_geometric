from typing import Optional

import torch
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.inits import reset
from torch_geometric.utils import softmax


class AttentionalAggregation(Aggregation):
    r"""The soft attention aggregation layer from the `"Graph Matching Networks
    for Learning the Similarity of Graph Structured Objects"
    <https://arxiv.org/abs/1904.12787>`_ paper.

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathrm{softmax} \left(
        h_{\mathrm{gate}} ( \mathbf{x}_n ) \right) \cdot
        h_{\mathbf{\Theta}} ( \mathbf{x}_n ),

    where :math:`h_{\mathrm{gate}} \colon \mathbb{R}^F \to
    \mathbb{R}` and :math:`h_{\mathbf{\Theta}}` denote neural networks, *i.e.*
    MLPs.

    Args:
        gate_nn (torch.nn.Module): A neural network :math:`h_{\mathrm{gate}}`
            that computes attention scores by mapping node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, 1]` (for
            node-level gating) or :obj:`[1, out_channels]` (for feature-level
            gating), *e.g.*, defined by :class:`torch.nn.Sequential`.
        nn (torch.nn.Module, optional): A neural network
            :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, out_channels]`
            before combining them with the attention scores, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
    """
    def __init__(
        self,
        gate_nn: torch.nn.Module,
        nn: Optional[torch.nn.Module] = None,
    ):
        super().__init__()

        from torch_geometric.nn import MLP

        self.gate_nn = self.gate_mlp = None
        if isinstance(gate_nn, MLP):
            self.gate_mlp = gate_nn
        else:
            self.gate_nn = gate_nn

        self.nn = self.mlp = None
        if isinstance(nn, MLP):
            self.mlp = nn
        else:
            self.nn = nn

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.gate_mlp)
        reset(self.nn)
        reset(self.mlp)

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        self.assert_two_dimensional_input(x, dim)

        if self.gate_mlp is not None:
            gate = self.gate_mlp(x, batch=index, batch_size=dim_size)
        else:
            gate = self.gate_nn(x)

        if self.mlp is not None:
            x = self.mlp(x, batch=index, batch_size=dim_size)
        elif self.nn is not None:
            x = self.nn(x)

        gate = softmax(gate, index, ptr, dim_size, dim)
        return self.reduce(gate * x, index, ptr, dim_size, dim)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'gate_nn={self.gate_mlp or self.gate_nn}, '
                f'nn={self.mlp or self.nn})')
