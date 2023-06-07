from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.inits import reset
from torch_geometric.utils import scatter


class ResNetPotential(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 num_layers: List[int]):

        super().__init__()
        sizes = [in_channels] + num_layers + [out_channels]
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(in_size, out_size),
                                torch.nn.LayerNorm(out_size), torch.nn.Tanh())
            for in_size, out_size in zip(sizes[:-2], sizes[1:-1])
        ])
        self.layers.append(torch.nn.Linear(sizes[-2], sizes[-1]))

        self.res_trans = torch.nn.ModuleList([
            torch.nn.Linear(in_channels, layer_size)
            for layer_size in num_layers + [out_channels]
        ])

    def forward(self, x: Tensor, y: Tensor, index: Optional[Tensor]) -> Tensor:
        if index is None:
            inp = torch.cat([x, y.expand(x.size(0), -1)], dim=1)
        else:
            inp = torch.cat([x, y[index]], dim=1)

        h = inp
        for layer, res in zip(self.layers, self.res_trans):
            h = layer(h)
            h = res(inp) + h

        if index is None:
            return h.mean()

        size = int(index.max().item() + 1)
        return scatter(h, index, dim=0, dim_size=size, reduce='mean').sum()


class MomentumOptimizer(torch.nn.Module):
    r"""
    Provides an inner loop optimizer for the implicitly defined output
    layer. It is based on an unrolled Nesterov momentum algorithm.

    Args:
        learning_rate (flaot): learning rate for optimizer.
        momentum (float): momentum for optimizer.
        learnable (bool): If :obj:`True` then the :obj:`learning_rate` and
            :obj:`momentum` will be learnable parameters. If False they
            are fixed. (default: :obj:`True`)
    """
    def __init__(self, learning_rate: float = 0.1, momentum: float = 0.9,
                 learnable: bool = True):
        super().__init__()

        self._initial_lr = learning_rate
        self._initial_mom = momentum
        self._lr = torch.nn.Parameter(Tensor([learning_rate]),
                                      requires_grad=learnable)
        self._mom = torch.nn.Parameter(Tensor([momentum]),
                                       requires_grad=learnable)
        self.softplus = torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()

    def reset_parameters(self):
        self._lr.data.fill_(self._initial_lr)
        self._mom.data.fill_(self._initial_mom)

    @property
    def learning_rate(self):
        return self.softplus(self._lr)

    @property
    def momentum(self):
        return self.sigmoid(self._mom)

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        index: Optional[Tensor],
        func: Callable[[Tensor, Tensor, Optional[Tensor]], Tensor],
        iterations: int = 5,
    ) -> Tuple[Tensor, float]:

        momentum_buffer = torch.zeros_like(y)
        for _ in range(iterations):
            val = func(x, y, index)
            grad = torch.autograd.grad(val, y, create_graph=True,
                                       retain_graph=True)[0]
            delta = self.learning_rate * grad
            momentum_buffer = self.momentum * momentum_buffer - delta
            y = y + momentum_buffer
        return y


class EquilibriumAggregation(Aggregation):
    r"""The equilibrium aggregation layer from the `"Equilibrium Aggregation:
    Encoding Sets via Optimization" <https://arxiv.org/abs/2202.12795>`_ paper.
    The output of this layer :math:`\mathbf{y}` is defined implicitly via a
    potential function :math:`F(\mathbf{x}, \mathbf{y})`, a regularization term
    :math:`R(\mathbf{y})`, and the condition

    .. math::
        \mathbf{y} = \min_\mathbf{y} R(\mathbf{y}) + \sum_{i}
        F(\mathbf{x}_i, \mathbf{y}).

    The given implementation uses a ResNet-like model for the potential
    function and a simple :math:`L_2` norm :math:`R(\mathbf{y}) =
    \textrm{softplus}(\lambda) \cdot {\| \mathbf{y} \|}^2_2` for the
    regularizer with learnable weight :math:`\lambda`.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_layers (List[int): List of hidden channels in the potential
            function.
        grad_iter (int): The number of steps to take in the internal gradient
            descent. (default: :obj:`5`)
        lamb (float): The initial regularization constant.
            (default: :obj:`0.1`)
    """
    def __init__(self, in_channels: int, out_channels: int,
                 num_layers: List[int], grad_iter: int = 5, lamb: float = 0.1):
        super().__init__()

        self.potential = ResNetPotential(in_channels + out_channels, 1,
                                         num_layers)
        self.optimizer = MomentumOptimizer()
        self.initial_lamb = lamb
        self.lamb = torch.nn.Parameter(Tensor(1), requires_grad=True)
        self.softplus = torch.nn.Softplus()
        self.grad_iter = grad_iter
        self.output_dim = out_channels
        self.reset_parameters()

    def reset_parameters(self):
        self.lamb.data.fill_(self.initial_lamb)
        reset(self.optimizer)
        reset(self.potential)

    def init_output(self, dim_size: int) -> Tensor:
        return torch.zeros(dim_size, self.output_dim, requires_grad=True,
                           device=self.lamb.device).float()

    def reg(self, y: Tensor) -> Tensor:
        return self.softplus(self.lamb) * y.square().sum(dim=-1).mean()

    def energy(self, x: Tensor, y: Tensor, index: Optional[Tensor]):
        return self.potential(x, y, index) + self.reg(y)

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        self.assert_index_present(index)

        dim_size = int(index.max()) + 1 if dim_size is None else dim_size

        with torch.enable_grad():
            y = self.optimizer(x, self.init_output(dim_size), index,
                               self.energy, iterations=self.grad_iter)

        return y

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}()')
