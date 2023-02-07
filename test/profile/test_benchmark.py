import torch

from torch_geometric.profile import benchmark
from torch_geometric.testing import withPackage


@withPackage('tabulate')
def test_benchmark():
    def add(x, y):
        return x + y

    benchmark(
        funcs=[add],
        args=(torch.randn(10), torch.randn(10)),
        num_steps=1,
        num_warmups=1,
        backward=True,
    )
