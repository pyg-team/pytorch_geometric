import random
import time

import torch
from torch import Tensor

import torch_geometric
from torch_geometric.nn import GraphSAGE, SAGEConv
from torch_geometric.testing import (
    disableExtensions,
    get_random_edge_index,
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
@disableExtensions()
@withPackage('torch>=2.0.0')
def test_dynamic_torch_compile(device):
    # torch._dynamo.config.specialize_int = False
    # torch._dynamo.config.verbose = True
    # torch._dynamo.config.verbose = True
    if device == torch.device('cpu'):
        return
    # conv = MySAGEConv(16, 16).to(device)
    conv = GraphSAGE(16, 16, num_layers=3).to(device)
    compiled_conv = torch_geometric.compile(conv, dynamic=True)

    t_total = 0
    for i in range(1000):
        N = random.randrange(50, 200)
        E = random.randrange(1000, 2000)

        x = torch.randn(N, 16, device=device)
        edge_index = get_random_edge_index(N, N, num_edges=E, device=device)

        if i > 100:
            torch.cuda.synchronize()
            t = time.perf_counter()
        # expected = conv(x, edge_index)
        out = compiled_conv(x, edge_index)

        if i > 100:
            torch.cuda.synchronize()
            t_total += time.perf_counter() - t
        # assert torch.allclose(out, expected, atol=1e-6)
    print(t_total)
