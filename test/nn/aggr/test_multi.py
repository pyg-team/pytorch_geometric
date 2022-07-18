import pytest
import torch

from torch_geometric.nn import MultiAggregation


@pytest.mark.parametrize('multi_aggr_tuple', [
    (dict(combine_mode='cat'), 3),
    (dict(combine_mode='sum'), 1),
    (dict(combine_mode='add'), 1),
    (dict(combine_mode='mul'), 1),
    (dict(combine_mode='mean'), 1),
    (dict(combine_mode='min'), 1),
    (dict(combine_mode='max'), 1),
    (dict(combine_mode='proj', in_channels=16), 1),
])
def test_multi_aggr(multi_aggr_tuple):
    multi_aggr_kwargs, expand = multi_aggr_tuple
    x = torch.randn(6, 16)
    index = torch.tensor([0, 0, 1, 1, 1, 2])
    ptr = torch.tensor([0, 2, 5, 6])

    aggrs = ['mean', 'sum', 'max']
    aggr = MultiAggregation(aggrs, **multi_aggr_kwargs)
    assert str(aggr) == (
        'MultiAggregation([\n'
        '  MeanAggregation(),\n'
        '  SumAggregation(),\n'
        '  MaxAggregation()\n'
        f"], combine_mode={multi_aggr_kwargs['combine_mode']})")

    out = aggr(x, index)
    assert torch.allclose(out, aggr(x, ptr=ptr))
    assert out.size() == (3, expand * x.size(1))
