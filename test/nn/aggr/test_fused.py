import pytest
import torch

from torch_geometric.nn.aggr.fused import FusedAggregation
from torch_geometric.nn.resolver import aggregation_resolver


@pytest.mark.parametrize('aggrs', [
    ['sum', 'mean', 'min', 'max', 'mul', 'var', 'std'],
    ['sum', 'min', 'max', 'mul', 'var', 'std'],
    ['min', 'max', 'mul', 'var', 'std'],
    ['mean', 'min', 'max', 'mul', 'var', 'std'],
    ['sum', 'min', 'max', 'mul', 'std'],
    ['mean', 'min', 'max', 'mul', 'std'],
    ['min', 'max', 'mul', 'std'],
])
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
    assert torch.allclose(out, expected)

    out.mean().backward()
    assert x.grad is not None
    expected.mean().backward()
    assert y.grad is not None
    assert torch.allclose(x.grad, y.grad)


if __name__ == '__main__':
    import argparse
    import time
    import warnings

    warnings.filterwarnings('ignore', '.*is in beta and the API may change.*')

    parser = argparse.ArgumentParser()
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    num_nodes, num_edges, num_feats = 1000, 50000, 64
    num_warmups, num_steps = 500, 1000

    aggrs = ['sum', 'mean', 'max', 'std']
    print(f'Aggregators: {", ".join(aggrs)}')
    print('=========================')
    aggrs = [aggregation_resolver(aggr) for aggr in aggrs]
    fused_aggr = FusedAggregation(aggrs)

    index = torch.randint(num_nodes, (num_edges, ), device='cuda')
    out_grad = torch.randn(num_nodes, len(aggrs) * num_feats, device='cuda')

    t_forward = t_backward = 0
    for i in range(num_warmups + num_steps):
        x = torch.randn(num_edges, num_feats, device='cuda')
        if args.backward:
            x.requires_grad_(True)
        torch.cuda.synchronize()

        t_start = time.perf_counter()
        outs = [aggr(x, index, dim_size=num_nodes) for aggr in aggrs]
        out = torch.cat(outs, dim=-1)

        torch.cuda.synchronize()
        if i >= num_warmups:
            t_forward += time.perf_counter() - t_start

        if args.backward:
            t_start = time.perf_counter()
            out.backward(out_grad)

            torch.cuda.synchronize()
            if i >= num_warmups:
                t_backward += time.perf_counter() - t_start

    print(f'Vanilla forward:  {t_forward:.4f}s')
    if args.backward:
        print(f'Vanilla backward: {t_backward:.4f}s')
    print('=========================')

    t_forward = t_backward = 0
    for i in range(num_warmups + num_steps):
        x = torch.randn(num_edges, num_feats, device='cuda')
        if args.backward:
            x.requires_grad_(True)
        torch.cuda.synchronize()

        t_start = time.perf_counter()
        out = torch.cat(fused_aggr(x, index, dim_size=num_nodes), dim=-1)

        torch.cuda.synchronize()
        if i >= num_warmups:
            t_forward += time.perf_counter() - t_start

        if args.backward:
            t_start = time.perf_counter()
            out.backward(out_grad)

            torch.cuda.synchronize()
            if i >= num_warmups:
                t_backward += time.perf_counter() - t_start

    print(f'Fused forward:    {t_forward:.4f}s')
    if args.backward:
        print(f'Fused backward:   {t_backward:.4f}s')
