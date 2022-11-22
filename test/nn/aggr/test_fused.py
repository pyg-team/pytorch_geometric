import torch

from torch_geometric.nn.aggr.fused import FusedAggregation
from torch_geometric.nn.resolver import aggregation_resolver


def test_fused_aggregation():
    x = torch.randn(6, 1)
    y = x.clone()

    x.requires_grad_(True)
    y.requires_grad_(True)

    index = torch.tensor([0, 0, 1, 1, 1, 2])

    aggrs = [
        aggregation_resolver('sum'),
        aggregation_resolver('mean'),
        aggregation_resolver('min'),
        aggregation_resolver('max'),
        aggregation_resolver('mul'),
        aggregation_resolver('var'),
        aggregation_resolver('std'),
    ]

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
