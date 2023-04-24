import torch

from torch_geometric.nn import WeightedMedianAggregation


def test_weighted_median():
    x = torch.randn(4, 8)
    weight = torch.tensor([0.1, 0.2, 0.3, 0.4])
    index = torch.tensor([0, 0, 0, 0])

    aggr = WeightedMedianAggregation()
    assert str(aggr) == 'WeightedMedianAggregation()'

    out = aggr(x, weight, index)
    assert out.size() == (1, 8)
