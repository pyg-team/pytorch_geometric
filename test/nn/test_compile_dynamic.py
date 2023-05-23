import random

import torch
from torch import Tensor

import torch_geometric
from torch_geometric.testing import (
    disableExtensions,
    get_random_edge_index,
    onlyFullTest,
    onlyLinux,
    withCUDA,
    withPackage,
)
from torch_geometric.utils import scatter


class MySAGEConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin_src = torch.nn.Linear(in_channels, out_channels)
        self.lin_dst = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x_j = x[edge_index[0]]
        out = scatter(x_j, edge_index[1], dim_size=x.size(0), reduce='mean')
        return self.lin_src(out) + self.lin_dst(x)


@withCUDA
@onlyLinux
@onlyFullTest
@disableExtensions
@withPackage('torch>2.0.0')
def test_dynamic_torch_compile(device):
    conv = MySAGEConv(64, 64).to(device)
    conv = torch_geometric.compile(conv, dynamic=True)

    optimizer = torch.optim.Adam(conv.parameters(), lr=0.01)

    for _ in range(10):
        N = random.randrange(100, 500)
        E = random.randrange(200, 1000)

        x = torch.randn(N, 64, device=device)
        edge_index = get_random_edge_index(N, N, E, device=device)

        optimizer.zero_grad()
        expected = conv(x, edge_index)
        expected.mean().backward()
        optimizer.step()
