import pytest
import torch

from torch_geometric.nn import (
    MaxAggr,
    MeanAggr,
    MinAggr,
    PowerMeanAggr,
    SoftmaxAggr,
    StdAggr,
    SumAggr,
    VarAggr,
)


@pytest.mark.parametrize(
    'Aggr', [MeanAggr, SumAggr, MaxAggr, MinAggr, VarAggr, StdAggr])
def test_basic_aggr(Aggr):
    x = torch.randn(6, 16)
    index = torch.tensor([0, 0, 1, 1, 1, 2])
    ptr = torch.tensor([0, 2, 5, 6])

    aggr = Aggr()
    assert str(aggr) == f'{Aggr.__name__}()'

    out = aggr(x, index)
    assert out.size() == (3, x.size(1))
    assert torch.allclose(out, aggr(x, ptr=ptr))


@pytest.mark.parametrize('Aggr', [SoftmaxAggr, PowerMeanAggr])
@pytest.mark.parametrize('learn', [True, False])
def test_gen_aggr(Aggr, learn):
    x = torch.randn(6, 16)
    index = torch.tensor([0, 0, 1, 1, 1, 2])
    ptr = torch.tensor([0, 2, 5, 6])

    aggr = Aggr(learn=learn)
    assert str(aggr) == f'{Aggr.__name__}()'

    out = aggr(x, index)
    assert out.size() == (3, x.size(1))
    assert torch.allclose(out, aggr(x, ptr=ptr))

    if learn:
        out.mean().backward()
        for param in aggr.parameters():
            assert not torch.isnan(param.grad).any()
