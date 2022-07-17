import pytest
import torch

from torch_geometric.nn import MultiAggregation


@pytest.mark.parametrize('combine_mode', [
    'cat',
    'sum',
    'add',
    'mul',
    'mean',
    'min',
    'max',
    'proj',
])
def test_multi_aggr(combine_mode):
    x = torch.randn(6, 16)
    index = torch.tensor([0, 0, 1, 1, 1, 2])
    ptr = torch.tensor([0, 2, 5, 6])

    aggrs = ['mean', 'sum', 'max']
    in_channels = 16 if combine_mode == 'proj' else None
    aggr = MultiAggregation(aggrs, combine_mode=combine_mode,
                            in_channels=in_channels)
    assert str(aggr) == ('MultiAggregation([\n'
                         '  MeanAggregation(),\n'
                         '  SumAggregation(),\n'
                         '  MaxAggregation()\n'
                         f'], combine_mode={combine_mode})')

    out = aggr(x, index)
    assert torch.allclose(out, aggr(x, ptr=ptr))

    expand = len(aggrs) if combine_mode == 'cat' else 1
    assert out.size() == (3, expand * x.size(1))
