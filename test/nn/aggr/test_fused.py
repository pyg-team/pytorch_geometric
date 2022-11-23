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
    out = aggr(x, index)

    expected = torch.cat([aggr(y, index) for aggr in aggrs], dim=-1)
    assert torch.allclose(out, expected)

    out.mean().backward()
    assert x.grad is not None
    expected.mean().backward()
    assert y.grad is not None
    assert torch.allclose(x.grad, y.grad)


if __name__ == '__main__':
    import time

    x = torch.randn(50000, 64, device='cuda')
    index = torch.randint(1000, (x.size(0), ), device='cuda')

    aggrs = ['sum', 'mean', 'max', 'std']
    aggrs = [aggregation_resolver(aggr) for aggr in aggrs]
    fused_aggr = FusedAggregation(aggrs)

    num_warmups, num_steps = (500, 1000)

    for i in range(num_warmups + num_steps):
        if i == num_warmups:
            torch.cuda.synchronize()
            t = time.perf_counter()
        torch.cat([aggr(x, index, dim_size=1000) for aggr in aggrs], dim=-1)
    torch.cuda.synchronize()
    print(f'Vanilla implementation: {time.perf_counter() - t:.4f} seconds')

    for i in range(num_warmups + num_steps):
        if i == num_warmups:
            torch.cuda.synchronize()
            t = time.perf_counter()
        fused_aggr(x, index, dim_size=1000)
    torch.cuda.synchronize()
    print(f'Fused implementation:   {time.perf_counter() - t:.4f} seconds')
