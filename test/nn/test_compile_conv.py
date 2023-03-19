import pytest
import torch

from torch_geometric import enable_compile
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.profile import benchmark
from torch_geometric.testing import onlyLinux, withCUDA, withPackage


@withCUDA
@onlyLinux
@enable_compile()
@withPackage('torch>=2.0.0')
@pytest.mark.parametrize('Conv', [GCNConv, SAGEConv, GATConv])
def test_compile_conv(device, Conv):
    x = torch.randn(10, 16, device=device)
    edge_index = torch.randint(0, x.size(0), (2, 40), device=device)

    conv = Conv(16, 32).to(device)
    out = torch.compile(conv)(x, edge_index)
    if not x.is_cuda or Conv != GATConv:  # TODO Fix compiled GATConv on GPU.
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

    for Conv in [GCNConv, SAGEConv, GATConv]:
        print(f'Conv: {Conv.__name__}')

        conv = Conv(64, 64).to(args.device)

        benchmark(
            funcs=[conv, torch.compile(enable_compile()(conv))],
            func_names=['Vanilla', 'Compiled'],
            args=(x, edge_index),
            num_steps=50 if args.device == 'cpu' else 500,
            num_warmups=10 if args.device == 'cpu' else 100,
            backward=args.backward,
        )
