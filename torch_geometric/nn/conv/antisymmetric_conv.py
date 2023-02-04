import math
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import GCNConv, MessagePassing
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import Adj


class AntiSymmetricConv(torch.nn.Module):
    r"""The anti-symmetric graph convolutional operator from the
    `"Anti-Symmetric DGN: a stable architecture for Deep Graph Networks"
    <https://openreview.net/forum?id=J3Y7cgZOOS>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{x}_i + \epsilon \cdot \sigma \left(
            (\mathbf{W}-\mathbf{W}^T-\gamma \mathbf{I}) \mathbf{x}_i +
            \Phi(\mathbf{X}, \mathcal{N}_i) + \mathbf{b}\right),

    where :math:`\Phi(\mathbf{X}, \mathcal{N}_i)` denotes a
    :class:`~torch.nn.conv.MessagePassing` layer.

    Args:
        in_channels (int): Size of each input sample.
        phi (MessagePassing, optional): The message passing module
            :math:`\Phi`. If set to :obj:`None`, will use a
            :class:`~torch_geometric.nn.conv.GCNConv` layer as default.
            (default: :obj:`None`)
        num_iters (int, optional): The number of times the anti-symmetric deep
            graph network operator is called. (default: :obj:`1`)
        epsilon (float, optional): The discretization step size
            :math:`\epsilon`. (default: :obj:`0.1`)
        gamma (float, optional): The strength of the diffusion :math:`\gamma`.
            It regulates the stability of the method. (default: :obj:`0.1`)
        act (str, optional): The non-linear activation function :math:`\sigma`,
            *e.g.*, :obj:`"tanh"` or :obj:`"relu"`. (default: :class:`"tanh"`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{in})`
    """
    def __init__(
        self,
        in_channels: int,
        phi: Optional[MessagePassing] = None,
        num_iters: int = 1,
        epsilon: float = 0.1,
        gamma: float = 0.1,
        act: Union[str, Callable, None] = 'tanh',
        act_kwargs: Optional[Dict[str, Any]] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_iters = num_iters
        self.gamma = gamma
        self.epsilon = epsilon
        self.act = activation_resolver(act, **(act_kwargs or {}))

        if phi is None:
            phi = GCNConv(in_channels, in_channels, bias=False)

        self.W = Parameter(torch.Tensor(in_channels, in_channels))
        self.register_buffer('eye', torch.eye(in_channels))
        self.phi = phi

        if bias:
            self.bias = Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.phi.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, *args, **kwargs) -> Tensor:
        """"""
        antisymmetric_W = self.W - self.W.t() - self.gamma * self.eye

        for _ in range(self.num_iters):
            h = self.phi(x, edge_index, *args, **kwargs)
            h = x @ antisymmetric_W.t() + h

            if self.bias is not None:
                h += self.bias

            if self.act is not None:
                h = self.act(h)

            x = x + self.epsilon * h

        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'{self.in_channels}, '
                f'phi={self.phi}, '
                f'num_iters={self.num_iters}, '
                f'epsilon={self.epsilon}, '
                f'gamma={self.gamma})')
