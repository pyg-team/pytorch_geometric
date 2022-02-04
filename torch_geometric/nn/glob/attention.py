from typing import Optional

import torch
from torch import Tensor
from torch_scatter import scatter_add

from torch_geometric.utils import softmax

from ..inits import reset


class GlobalAttention(torch.nn.Module):
    r"""Global soft attention layer from the `"Gated Graph Sequence Neural
    Networks" <https://arxiv.org/abs/1511.05493>`_ paper

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathrm{softmax} \left(
        h_{\mathrm{gate}} ( \mathbf{x}_n ) \right) \odot
        h_{\mathbf{\Theta}} ( \mathbf{x}_n ),

    where :math:`h_{\mathrm{gate}} \colon \mathbb{R}^F \to
    \mathbb{R}` and :math:`h_{\mathbf{\Theta}}` denote neural networks, *i.e.*
    MLPs.

    Args:
        gate_nn (torch.nn.Module): A neural network :math:`h_{\mathrm{gate}}`
            that computes attention scores by mapping node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, 1]`, *e.g.*,
            defined by :class:`torch.nn.Sequential`.
        nn (torch.nn.Module, optional): A neural network
            :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, out_channels]`
            before combining them with the attention scores, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)`,
          batch vector :math:`(|\mathcal{V}|)` *(optional)*
        - **output:**
          graph features :math:`(|\mathcal{G}|, 2 * F)` where
          :math:`|\mathcal{G}|` denotes the number of graphs in the batch
    """
    def __init__(self, gate_nn: torch.nn.Module,
                 nn: Optional[torch.nn.Module] = None):
        super().__init__()
        self.gate_nn = gate_nn
        self.nn = nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, x: Tensor, batch: Optional[Tensor] = None,
                size: Optional[int] = None) -> Tensor:
        r"""
        Args:
            x (Tensor): The input node features.
            batch (LongTensor, optional): A vector that maps each node to its
                respective graph identifier. (default: :obj:`None`)
            size (int, optional): The number of graphs in the batch.
                (default: :obj:`None`)
        """
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.int64)

        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = int(batch.max()) + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(gate_nn={self.gate_nn}, '
                f'nn={self.nn})')
