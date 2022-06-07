import torch

from torch_geometric.nn import MultiAggregation


def test_multi_aggr():
    x = torch.randn(6, 16)
    index = torch.tensor([0, 0, 1, 1, 1, 2])
    ptr = torch.tensor([0, 2, 5, 6])

    aggrs = ['mean', 'sum', 'max']
    aggr = MultiAggregation(aggrs)
    assert str(aggr) == ('MultiAggregation([\n'
                         '  MeanAggregation(),\n'
                         '  SumAggregation(),\n'
                         '  MaxAggregation()\n'
                         '])')

    out = aggr(x, index)
    assert out.size() == (3, len(aggrs) * x.size(1))
    assert torch.allclose(out, aggr(x, ptr=ptr))
