from typing import Callable, List, Optional, Tuple

import torch
from torch_scatter import scatter

from torch_geometric.nn.inits import reset


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

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                batch: Optional[torch.Tensor]) -> torch.Tensor:
        if batch is None:
            inp = torch.cat([x, y.expand(x.size(0), -1)], dim=1)
        else:
            inp = torch.cat([x, y[batch]], dim=1)

        h = inp
        for layer, res in zip(self.layers, self.res_trans):
            h = layer(h)
            h = res(inp) + h

        if batch is None:
            return h.mean()

        size = int(batch.max().item() + 1)
        return scatter(x, batch, dim=0, dim_size=size, reduce='mean').sum()


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
        self._lr = torch.nn.Parameter(torch.Tensor([learning_rate]),
                                      requires_grad=learnable)
        self._mom = torch.nn.Parameter(torch.Tensor([momentum]),
                                       requires_grad=learnable)
        self.softplus = torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()
        self.register_full_backward_hook(self.gradient_hook)

    @staticmethod
    def gradient_hook(module, grad_input, grad_output):
        pass

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
        x: torch.Tensor,
        y: torch.Tensor,
        batch: Optional[torch.Tensor],
        func: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
                       torch.Tensor],
        iterations: int = 5,
    ) -> Tuple[torch.Tensor, float]:

        momentum = torch.zeros_like(y)
        for _ in range(iterations):
            val = func(x, y, batch)
            grad = torch.autograd.grad(val, y, create_graph=True,
                                       retain_graph=True)[0]
            momentum = self.momentum * momentum - self.learning_rate * grad
            y = y + momentum
        return y


class EquilibriumAggregation(torch.nn.Module):
    r"""
    The graph global pooling layer from the
    `"Equilibrium Aggregation: Encoding Sets via Optimization"
    <https://arxiv.org/abs/2202.12795>`_ paper.
    This output of this layer :math:`\mathbf{y}` is defined implicitly by
    defining a potential function :math:`F(\mathbf{x}, \mathbf{y})`
    and regulatization function :math:`R(\mathbf{y})` and the condition

    .. math::
        \mathbf{y} = \min_\mathbf{y} R(\mathbf{y}) +
        \sum_{i} F(\mathbf{x}_i, \mathbf{y})

    This implementation use a ResNet Like model for the potential function
    and a simple L2 norm for the regularizer with learnable weight
    :math:`\lambda`.

    .. note::

        The forward function of this layer accepts a :obj:`batch` argument that
        works like the other global pooling layers when working on input that
        is from multiple graphs.

    Args:
        in_channels (int): The number of channels in the input to the layer.
        out_channels (float): The number of channels in the ouput.
        num_layers (List[int): A list of the number of hidden units in the
            potential function.
        grad_iter (int): The number of steps to take in the internal gradient
            descent. (default: :obj:`5`)
        lamb (float): The initial regularization constant. Is learnable.
            descent. (default: :obj:`0.1`)
    """
    def __init__(self, in_channels: int, out_channels: int,
                 num_layers: List[int], grad_iter: int = 5, lamb: float = 0.1):
        super().__init__()

        self.potential = ResNetPotential(in_channels + out_channels, 1,
                                         num_layers)
        self.optimizer = MomentumOptimizer()
        self._initial_lambda = lamb
        self._labmda = torch.nn.Parameter(torch.Tensor([lamb]),
                                          requires_grad=True)
        self.softplus = torch.nn.Softplus()
        self.grad_iter = grad_iter
        self.output_dim = out_channels
        self.reset_parameters()

    def reset_parameters(self):
        self.lamb.data.fill_(self._initial_lambda)
        reset(self.optimizer)
        reset(self.potential)

    @property
    def lamb(self):
        return self.softplus(self._labmda)

    def init_output(self,
                    batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = 1 if batch is None else int(batch.max().item() + 1)
        return torch.zeros(batch_size, self.output_dim,
                           requires_grad=True).float()

    def reg(self, y: torch.Tensor) -> float:
        return self.lamb * y.square().mean(dim=1).sum(dim=0)

    def energy(self, x: torch.Tensor, y: torch.Tensor,
               batch: Optional[torch.Tensor]):
        return self.potential(x, y, batch) + self.reg(y)

    def forward(self, x: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:

        with torch.enable_grad():
            y = self.optimizer(x, self.init_output(batch), batch, self.energy,
                               iterations=self.grad_iter)

        return y

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}()')
