from typing import List

import torch

EPS = 1e-15


class ResNetPotential(torch.nn.Module):
    def __init__(self, input_size: int, layers: List[int]):

        super().__init__()
        output_size = 1
        sizes = [input_size] + layers + [output_size]
        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(in_size, out_size),
                                torch.nn.LayerNorm(out_size), torch.nn.Tanh())
            for in_size, out_size in zip(sizes[:-1], sizes[1:])
        ])

        self.res_trans = torch.nn.ModuleList([
            torch.nn.Linear(input_size, layer_size)
            for layer_size in layers + [output_size]
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer, res in zip(self.layers, self.res_trans):
            h = layer(h)
            h = res(x) + h

        return h


class EquilibriumAggregation(torch.nn.Module):
    """
    Args:
        potential (torch.nn.Module): trainable potenial function
    """
    def __init__(self, input_dim: int, output_dim: int, layers: List[int],
                 grad_iter: int = 5, alpha: float = 0.1):
        super().__init__()

        self.potential = ResNetPotential(input_dim + output_dim, layers)
        self.lamb = torch.nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.grad_iter = grad_iter
        self.alpha = alpha
        self.output_dim = output_dim

    def init_output(self):
        return torch.zeros(self.output_dim, requires_grad=True)

    def reg(self, y):
        return torch.nn.Softplus()(
            self.lamb) * (y + EPS).square().mean().sqrt()

    def combine_input(self, x, y):
        return torch.cat([x, y.expand(x.size(0), -1)], dim=1)

    def forward(self, x: torch.Tensor):
        grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        yhat = self.init_output()
        for _ in range(self.grad_iter):
            erg = self.potential(self.combine_input(
                x, yhat)).mean() + self.reg(yhat)
            yhat = yhat - self.alpha * torch.autograd.grad(
                erg, yhat, create_graph=True, retain_graph=True)[0]
        torch.set_grad_enabled(grad_enabled)
        return yhat

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}()')
