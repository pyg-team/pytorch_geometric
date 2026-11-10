import torch

from torch_geometric.nn import GRUAggregation


def test_gru_aggregation():
    x = torch.randn(6, 16)
    index = torch.tensor([0, 0, 1, 1, 1, 2])

    aggr = GRUAggregation(16, 32)
    assert str(aggr) == 'GRUAggregation(16, 32)'

    out = aggr(x, index)
    assert out.size() == (3, 32)


def test_gru_aggregation_empty_input():
    aggr = GRUAggregation(16, 32)

    out = aggr(torch.empty(0, 16), torch.empty(0, dtype=torch.long),
               dim_size=3)
    assert out.size() == (3, 32)
    assert torch.isfinite(out).all()
