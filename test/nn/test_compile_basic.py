import torch

import torch_geometric
from torch_geometric.profile import benchmark
from torch_geometric.testing import (
    disableExtensions,
    onlyLinux,
    withCUDA,
    withPackage,
)
from torch_geometric.utils import scatter


# Basic "Gather-Apply-Scatter" patterns commonly used in PyG:
def gather_scatter(x, edge_index, reduce='sum'):
    row, col = edge_index
    x_j = x[row]
    return scatter(x_j, col, dim_size=x.size(0), reduce=reduce)


def gather_cat_scatter(x, edge_index, reduce='sum'):
    row, col = edge_index
    x_ij = torch.cat([x[col], x[row]], dim=-1)
    return scatter(x_ij, col, dim_size=x.size(0), reduce=reduce)


def gather_weight_scatter(x, edge_index, edge_weight, reduce='sum'):
    row, col = edge_index
    x_j = x[row] * edge_weight.view(-1, 1)
    return scatter(x_j, col, dim_size=x.size(0), reduce=reduce)


def gather_transform_scatter(x, edge_index, matrix, reduce='sum'):
    row, col = edge_index
    x_j = x[row] @ matrix
    return scatter(x_j, col, dim_size=x.size(0), reduce=reduce)


def fused_gather_scatter(x, edge_index, reduce=['sum', 'mean', 'max']):
    row, col = edge_index
    x_j = x[row]
    outs = [scatter(x_j, col, dim_size=x.size(0), reduce=r) for r in reduce]
    return torch.cat(outs, dim=-1)


@withCUDA
@onlyLinux
@disableExtensions()
@withPackage('torch>=2.0.0')
def test_torch_compile(device):
    x = torch.randn(10, 16, device=device)
    edge_index = torch.randint(0, x.size(0), (2, 40), device=device)
    edge_weight = torch.rand(edge_index.size(1), device=device)
    matrix = torch.randn(x.size(-1), x.size(-1), device=device)

    expected = gather_scatter(x, edge_index)
    compiled_op = torch_geometric.compile(gather_scatter)
    out = compiled_op(x, edge_index)
    assert torch.allclose(out, expected, atol=1e-6)

    expected = gather_cat_scatter(x, edge_index)
    compiled_op = torch_geometric.compile(gather_cat_scatter)
    out = compiled_op(x, edge_index)
    assert torch.allclose(out, expected, atol=1e-6)

    expected = gather_weight_scatter(x, edge_index, edge_weight)
    compiled_op = torch_geometric.compile(gather_weight_scatter)
    out = compiled_op(x, edge_index, edge_weight)
    assert torch.allclose(out, expected, atol=1e-6)

    expected = gather_transform_scatter(x, edge_index, matrix)
    compiled_op = torch_geometric.compile(gather_transform_scatter)
    out = compiled_op(x, edge_index, matrix)
    assert torch.allclose(out, expected, atol=1e-6)

    expected = fused_gather_scatter(x, edge_index)
    compiled_op = torch_geometric.compile(fused_gather_scatter)
    out = compiled_op(x, edge_index)
    assert torch.allclose(out, expected, atol=1e-6)


@withCUDA
@onlyLinux
@disableExtensions()
@withPackage('torch>=2.0.0')
def test_dynamic_torch_compile(device):
    compiled_gather_scatter = torch_geometric.compile(gather_scatter)

    x = torch.randn(10, 16, device=device)
    edge_index = torch.randint(0, x.size(0), (2, 40), device=device)

    expected = gather_scatter(x, edge_index)
    out = compiled_gather_scatter(x, edge_index)
    assert torch.allclose(out, expected, atol=1e-6)

    x = torch.randn(20, 16, device=device)
    edge_index = torch.randint(0, x.size(0), (2, 80), device=device)

    expected = gather_scatter(x, edge_index)
    out = compiled_gather_scatter(x, edge_index)
    assert torch.allclose(out, expected, atol=1e-6)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    num_nodes, num_edges = 10_000, 200_000
    x = torch.randn(num_nodes, 64, device=args.device)
    edge_index = torch.randint(num_nodes, (2, num_edges), device=args.device)
    edge_weight = torch.rand(num_edges, device=args.device)
    matrix = torch.randn(64, 64, device=args.device)

    for reduce in ['sum', 'mean', 'max']:
        print(f'Aggregator: {reduce}')

        benchmark(
            funcs=[
                gather_scatter,
                torch_geometric.compile(gather_scatter),
            ],
            func_names=['Vanilla', 'Compiled'],
            args=(x, edge_index, reduce),
            num_steps=50 if args.device == 'cpu' else 500,
            num_warmups=10 if args.device == 'cpu' else 100,
            backward=args.backward,
        )

        benchmark(
            funcs=[
                gather_cat_scatter,
                torch_geometric.compile(gather_cat_scatter),
            ],
            func_names=['Vanilla Cat', 'Compiled Cat'],
            args=(x, edge_index, reduce),
            num_steps=50 if args.device == 'cpu' else 500,
            num_warmups=10 if args.device == 'cpu' else 100,
            backward=args.backward,
        )

        benchmark(
            funcs=[
                gather_weight_scatter,
                torch_geometric.compile(gather_weight_scatter),
            ],
            func_names=['Vanilla Weight', 'Compiled Weight'],
            args=(x, edge_index, edge_weight, reduce),
            num_steps=50 if args.device == 'cpu' else 500,
            num_warmups=10 if args.device == 'cpu' else 100,
            backward=args.backward,
        )

        benchmark(
            funcs=[
                gather_transform_scatter,
                torch_geometric.compile(gather_transform_scatter),
            ],
            func_names=['Vanilla Transform', 'Compiled Transform'],
            args=(x, edge_index, matrix, reduce),
            num_steps=50 if args.device == 'cpu' else 500,
            num_warmups=10 if args.device == 'cpu' else 100,
            backward=args.backward,
        )

    benchmark(
        funcs=[
            fused_gather_scatter,
            torch_geometric.compile(fused_gather_scatter),
        ],
        func_names=['Vanilla Fused', 'Compiled Fused'],
        args=(x, edge_index),
        num_steps=50 if args.device == 'cpu' else 500,
        num_warmups=10 if args.device == 'cpu' else 100,
        backward=args.backward,
    )
