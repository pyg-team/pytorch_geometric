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
    index = torch.tensor([0, 0, 1, 1, 1, 2])

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
