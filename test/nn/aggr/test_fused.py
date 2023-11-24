import pytest
import torch

import torch_geometric
from torch_geometric.testing import withCUDA, disableExtensions, withPackage
from torch_geometric.nn.aggr.fused import FusedAggregation
from torch_geometric.nn.resolver import aggregation_resolver
from torch_geometric.profile import benchmark


aggrs_params = [
    ['sum', 'mean', 'min', 'max', 'mul', 'var', 'std'],
    ['sum', 'min', 'max', 'mul', 'var', 'std'],
    ['min', 'max', 'mul', 'var', 'std'],
    ['mean', 'min', 'max', 'mul', 'var', 'std'],
    ['sum', 'min', 'max', 'mul', 'std'],
    ['mean', 'min', 'max', 'mul', 'std'],
    ['min', 'max', 'mul', 'std'],
]


@pytest.mark.parametrize('aggrs', aggrs_params)
def test_fused_aggregation(aggrs):
    aggrs = [aggregation_resolver(aggr) for aggr in aggrs]

    x = torch.randn(6, 1)
    y = x.clone()
    index = torch.tensor([0, 0, 1, 1, 1, 3])

    x.requires_grad_(True)
    y.requires_grad_(True)

    aggr = FusedAggregation(aggrs)
    assert str(aggr) == 'FusedAggregation()'
    out = torch.cat(aggr(x, index), dim=-1)

    expected = torch.cat([aggr(y, index) for aggr in aggrs], dim=-1)
    assert torch.allclose(out, expected, atol=1e-5)

    jit = torch.jit.script(aggr)
    assert torch.allclose(torch.cat(jit(x, index), dim=-1), out, atol=1e-5)

    out.mean().backward()
    assert x.grad is not None
    expected.mean().backward()
    assert y.grad is not None
    assert torch.allclose(x.grad, y.grad, atol=1e-5)


@withCUDA
@disableExtensions
@withPackage('torch>=2.1.0')
@pytest.mark.parametrize('aggrs', aggrs_params)
def test_compile_fused_aggregation(aggrs, device):
    aggrs = [aggregation_resolver(aggr) for aggr in aggrs]

    x = torch.randn(6, 1, device=device)
    y = x.clone()
    index = torch.tensor([0, 0, 1, 1, 1, 3], device=device)

    aggr = FusedAggregation(aggrs).to(device)
    out1 = torch.cat(aggr(x, index), dim=-1)

    opt = torch_geometric.compile(aggr)
    out2 = torch.cat(opt(x, index), dim=-1)
    assert torch.allclose(out1, out2, atol=1e-7)



def test_empty_fused_std_aggregation():
    aggrs = [aggregation_resolver(aggr) for aggr in ['mean', 'var', 'std']]
    aggr = FusedAggregation(aggrs)

    x = torch.empty(0, 6).reshape(0, 6)
    index = torch.empty(0, dtype=torch.long)

    out = torch.cat(aggr(x, index, dim_size=5), dim=-1)
    assert out.size() == (5, 18)
    assert float(out.abs().sum()) == 0.0


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    num_nodes, num_edges = 1_000, 50_000
    x = torch.randn(num_edges, 64, device=args.device)
    index = torch.randint(num_nodes, (num_edges, ), device=args.device)

    aggrs = ['sum', 'mean', 'max', 'std']
    print(f'Aggregators: {", ".join(aggrs)}')

    aggrs = [aggregation_resolver(aggr) for aggr in aggrs]
    fused_aggregation = FusedAggregation(aggrs)

    def naive_aggr(x, index, dim_size):
        outs = [aggr(x, index, dim_size=dim_size) for aggr in aggrs]
        return torch.cat(outs, dim=-1)

    def fused_aggr(x, index, dim_size):
        outs = fused_aggregation(x, index, dim_size=dim_size)
        return torch.cat(outs, dim=-1)

    benchmark(
        funcs=[naive_aggr, fused_aggr],
        func_names=['Naive', 'Fused'],
        args=(x, index, num_nodes),
        num_steps=100 if args.device == 'cpu' else 1000,
        num_warmups=50 if args.device == 'cpu' else 500,
        backward=args.backward,
    )
