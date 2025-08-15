import pytest
import torch
from torch import Tensor

from torch_geometric import EdgeIndex
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.profile import benchmark
from torch_geometric.testing import (
    onlyFullTest,
    onlyLinux,
    withDevice,
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


@withDevice
@onlyLinux
@onlyFullTest
@withPackage('torch>=2.1.0')
@pytest.mark.parametrize('Conv', [GCNConv, SAGEConv])
def test_compile_conv(device, Conv):
    import torch._dynamo as dynamo

    x = torch.randn(10, 16, device=device)
    edge_index = torch.randint(0, x.size(0), (2, 40), device=device)

    if Conv == GCNConv:
        conv = Conv(16, 32, add_self_loops=False).to(device)
    else:
        conv = Conv(16, 32).to(device)

    explanation = dynamo.explain(conv)(x, edge_index)
    assert explanation.graph_break_count == 0

    out = torch.compile(conv)(x, edge_index)
    assert torch.allclose(conv(x, edge_index), out, atol=1e-6)


@withDevice
@onlyLinux
@onlyFullTest
@withPackage('torch==2.3')
@pytest.mark.parametrize('Conv', [GCNConv, SAGEConv])
def test_compile_conv_edge_index(device, Conv):
    import torch._dynamo as dynamo

    x = torch.randn(10, 16, device=device)
    edge_index = torch.randint(0, x.size(0), (2, 40), device=device)
    edge_index = EdgeIndex(edge_index, sparse_size=(10, 10))
    edge_index = edge_index.sort_by('col')[0]
    edge_index.fill_cache_()

    if Conv == GCNConv:
        conv = Conv(16, 32, normalize=False).to(device)
    else:
        conv = Conv(16, 32).to(device)

    explanation = dynamo.explain(conv)(x, edge_index)
    assert explanation.graph_break_count == 0

    out = torch.compile(conv, fullgraph=True)(x, edge_index)
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
        funcs=[conv, torch.compile(conv)],
        func_names=['Vanilla', 'Compiled'],
        args=(x, edge_index),
        num_steps=50 if args.device == 'cpu' else 500,
        num_warmups=10 if args.device == 'cpu' else 100,
        backward=args.backward,
    )

    for Conv in [GCNConv, SAGEConv]:
        print(f'Conv: {Conv.__name__}')

        conv = Conv(64, 64).to(args.device)
        compiled_conv = torch.compile(conv)

        benchmark(
            funcs=[conv, compiled_conv],
            func_names=['Vanilla', 'Compiled'],
            args=(x, edge_index),
            num_steps=50 if args.device == 'cpu' else 500,
            num_warmups=10 if args.device == 'cpu' else 100,
            backward=args.backward,
        )
