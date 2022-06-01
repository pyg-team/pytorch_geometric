from itertools import combinations

import pytest
import torch

from torch_geometric.nn import (
    LSTMAggregation,
    MultiAggregation,
    PowerMeanAggregation,
    SoftmaxAggregation,
)

aggr_list = [
    'mean', 'sum', 'max', 'min', 'mul', 'std', 'var',
    SoftmaxAggregation(learn=True),
    PowerMeanAggregation(learn=True),
    LSTMAggregation(16, 16)
]
aggrs = [list(aggr) for aggr in combinations(aggr_list, 3)] + \
    [[aggr] for aggr in aggr_list]


@pytest.mark.parametrize('aggrs', aggrs)
def test_my_multiple_aggr(aggrs):
    x = torch.randn(6, 16)
    index = torch.tensor([0, 0, 1, 1, 1, 2])
    aggr = MultiAggregation(aggrs=aggrs)
    out = aggr(x, index)
    assert out.size() == (3, len(aggrs) * 16)
