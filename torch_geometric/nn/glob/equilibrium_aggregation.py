from typing import List, Optional

import torch
from torch_scatter import scatter

from torch_geometric.nn.inits import reset


class FCResNetBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 num_layers: List[int]):

        super().__init__()
        sizes = [in_channels] + num_layers + [out_channels]
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(in_size, out_size),
                # torch.nn.LayerNorm(out_size),
                torch.nn.Tanh())
            for in_size, out_size in zip(sizes[:-1], sizes[1:])
        ])

        self.res_trans = torch.nn.ModuleList([
            torch.nn.Linear(in_channels, layer_size)
            for layer_size in num_layers + [out_channels]
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer, res in zip(self.layers, self.res_trans):
            h = layer(h)
            h = res(x) + h

        return h


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
        alpha (float): The step size of the internal gradient descent.
            (default: :obj:`0.1`)

    """
    def __init__(self, in_channels: int, out_channels: int,
                 num_layers: List[int], grad_iter: int = 5,
                 alpha: float = 0.1):
        super().__init__()

        self.potential = FCResNetBlock(in_channels + out_channels, 1,
                                       num_layers)
        self.lamb = torch.nn.Parameter(torch.Tensor([0.1]), requires_grad=True)
        self.softplus = torch.nn.Softplus()
        self.grad_iter = grad_iter
        self.alpha = alpha
        self.output_dim = out_channels

        self.reset_parameters()

    def reset_parameters(self):
        self.lamb.data.fill_(0.1)
        reset(self.potential)

    def init_output(self,
                    batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = 1 if batch is None else int(batch.max().item() + 1)
        return torch.zeros(batch_size, self.output_dim,
                           requires_grad=True).float() + 1e-15

    def reg(self, y: torch.Tensor) -> float:
        return self.softplus(self.lamb) * y.norm(2, dim=1, keepdim=True)

    def combine_input(self, x: torch.Tensor, y: torch.Tensor,
                      batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if batch is None:
            return torch.cat([x, y.expand(x.size(0), -1)], dim=1)
        return torch.cat([x, y[batch]], dim=1)

    def forward(self, x: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.enable_grad():
            yhat = self.init_output(batch)
            for _ in range(self.grad_iter):
                z = self.combine_input(x, yhat, batch)
                potential = self.potential(z)
                if batch is None:
                    potential = potential.mean(axis=0, keepdim=True)
                else:
                    size = int(batch.max().item() + 1)
                    potential = scatter(potential, batch, dim=0, dim_size=size,
                                        reduce='mean')
                reg = self.reg(yhat)
                enrg = (potential + reg).sum()
                yhat = yhat - self.alpha * torch.autograd.grad(
                    enrg, yhat, create_graph=True, retain_graph=True)[0]
            return yhat

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}()')
