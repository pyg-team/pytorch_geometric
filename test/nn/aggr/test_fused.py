import torch

from torch_geometric.nn.aggr.fused import FusedAggregation
from torch_geometric.nn.resolver import aggregation_resolver


def test_fused_aggregation():
    x = torch.randn(6, 1)
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

    expected = torch.cat([aggr(x, index) for aggr in aggrs], dim=-1)
    assert torch.allclose(out, expected)
