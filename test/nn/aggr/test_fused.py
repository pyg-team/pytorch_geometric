import torch

from torch_geometric.nn.aggr.fused import FusedAggregation
from torch_geometric.nn.resolver import aggregation_resolver


def test_fused_aggregation():
    aggr = FusedAggregation([
        aggregation_resolver('sum'),
        aggregation_resolver('mean'),
        aggregation_resolver('min'),
        aggregation_resolver('max'),
        aggregation_resolver('mul'),
        aggregation_resolver('var'),
        aggregation_resolver('std'),
    ])

    x = torch.randn(6, 1)
    index = torch.tensor([0, 0, 1, 1, 1, 2])

    out = aggr(x, index)
    print(out)
