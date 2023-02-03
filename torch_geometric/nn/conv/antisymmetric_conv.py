import math
from typing import Optional

import torch
from torch.nn import Module, Parameter, init

from torch_geometric.nn.conv import GCNConv
from torch_geometric.typing import Adj, OptTensor


class AntiSymmetricConv(Module):
    r"""The anti-symmetric deep graph network operator from the
    `"Anti-Symmetric DGN: a stable architecture for Deep Graph Networks"
    <https://openreview.net/forum?id=J3Y7cgZOOS>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{x}_i + \epsilon \sigma \left(
            (\mathbf{W}-\mathbf{W}^T-\gamma \mathbf{I}) \mathbf{x}_i +
            \Phi(\mathbf{X}, \mathcal{N}_i) + \mathbf{b}\right),

    where :math:`\Phi(\mathbf{X}, \mathcal{N}_i)` denotes the aggregation
    function for the states of the nodes in the neighborhood of :math:`i`.


    Args:
        in_channels (int): Size of each input sample.
        phi (torch.nn.Module, optional): the function that aggregates nodes
            (and edges) information :math:`\Phi`. If set to :obj:`None`, then
            the aggregator is instantiated as
            :obj:`GCNConv(in_channels, in_channels)`. (default :obj:`None`)
        num_iters (int, optional): the number of times the anti-symmetric deep
            graph network operator is called leveraging weight sharing.
            (default :obj:`1`)
        epsilon (float, optional): the discretization step size
            :math:`\epsilon`. (default :obj:`0.1`)
        gamma (float, optional): the strength of the diffusion :math:`\gamma`.
            It regulates the stability of the method. (default :obj:`0.1`)
        activ_fun (str, optional): the monotonically non-decreasing activation
            function :math:`\sigma`, e.g., :class:`torch.tanh`,
            :class:`torch.relu`. (default :class:`torch.tanh`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{in})`
    """
    def __init__(self, in_channels: int, phi: Optional[torch.nn.Module] = None,
                 num_iters: int = 1, epsilon: float = 0.1, gamma: float = 0.1,
                 activ_fun: str = 'tanh', bias: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.W = Parameter(torch.empty((in_channels, in_channels)),
                           requires_grad=True)
        self.bias = (Parameter(torch.empty(in_channels), requires_grad=True)
                     if bias else None)
        self.eye = Parameter(torch.eye(in_channels), requires_grad=False)
        self.phi = (GCNConv(in_channels, in_channels, bias=False)
                    if phi is None else phi)
        self.in_channels = in_channels
        self.num_iters = num_iters
        self.gamma = gamma
        self.epsilon = epsilon
        self.activation = getattr(torch, activ_fun)
        self.activ_fun = activ_fun
        self.b = bias

        self.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> torch.Tensor:
        """"""
        antisymmetric_W = self.W - self.W.T - self.gamma * self.eye

        for _ in range(self.num_iters):
            neigh_x = self.phi(x=x, edge_index=edge_index,
                               edge_weight=edge_weight)
            conv = x @ antisymmetric_W.T + neigh_x

            if self.bias is not None:
                conv += self.bias

            x = x + self.epsilon * self.activation(conv)
        return x

    def reset_parameters(self):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'phi={self.phi}, '
                f'num_iters={self.num_iters}, '
                f'epsilon={self.epsilon}, '
                f'gamma={self.gamma}, '
                f'activ_fun={self.activ_fun}, '
                f'bias={self.b})')
