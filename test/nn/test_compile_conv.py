import pytest
import torch
from torch import Tensor

import torch_geometric
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.profile import benchmark
from torch_geometric.testing import onlyLinux, withCUDA, withPackage
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
@withPackage('torch>=2.0.0')
@pytest.mark.parametrize('Conv', [GCNConv, SAGEConv])
def test_compile_conv(device, Conv):
    x = torch.randn(10, 16, device=device)
    edge_index = torch.randint(0, x.size(0), (2, 40), device=device)

    conv = Conv(16, 32).to(device)
    out = torch_geometric.compile(conv)(x, edge_index)
    assert torch.allclose(conv(x, edge_index), out, atol=1e-6)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    num_nodes, num_edges = 10_000, 200_000
    x = torch.randn(num_nodes, 64, device=args.device)
    edge_index = torch.randint(num_nodes, (2, num_edges), device=args.device)

    conv = MySAGEConv(64, 64).to(args.device)
    benchmark(
        funcs=[conv, torch_geometric.compile(conv)],
        func_names=['Vanilla', 'Compiled'],
        args=(x, edge_index),
        num_steps=50 if args.device == 'cpu' else 500,
        num_warmups=10 if args.device == 'cpu' else 100,
        backward=args.backward,
    )

    for Conv in [GCNConv, SAGEConv]:
        print(f'Conv: {Conv.__name__}')

        conv = Conv(64, 64).to(args.device)
        compiled_conv = torch_geometric.compile(conv)

        benchmark(
            funcs=[conv, compiled_conv],
            func_names=['Vanilla', 'Compiled'],
            args=(x, edge_index),
            num_steps=50 if args.device == 'cpu' else 500,
            num_warmups=10 if args.device == 'cpu' else 100,
            backward=args.backward,
        )
