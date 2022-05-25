import pytest
import torch

from torch_geometric.nn import (
    MaxAggr,
    MeanAggr,
    MinAggr,
    PowermeanAggr,
    SoftmaxAggr,
    StdAggr,
    SumAggr,
    VarAggr,
)


@pytest.mark.parametrize('aggr', [MeanAggr, MaxAggr, MinAggr, SumAggr])
def test_basic_aggr(aggr):
    src = torch.randn(6, 64)
    index = torch.tensor([0, 1, 0, 1, 2, 1])
    out = aggr()(src, index)
    assert out.shape[0] == index.unique().shape[0]


@pytest.mark.parametrize('aggr', [SoftmaxAggr, PowermeanAggr])
def test_gen_aggr(aggr):
    src = torch.randn(6, 64)
    index = torch.tensor([0, 1, 0, 1, 2, 1])
    for learn in [True, False]:
        if issubclass(aggr, SoftmaxAggr):
            aggregator = aggr(t=1, learn=learn)
        elif issubclass(aggr, PowermeanAggr):
            aggregator = aggr(p=1, learn=learn)
        else:
            raise NotImplementedError
        out = aggregator(src, index)
        if any(map(lambda x: x.requires_grad, aggregator.parameters())):
            out.mean().backward()
            for param in aggregator.parameters():
                assert not torch.isnan(param.grad).any()


@pytest.mark.parametrize('aggr', [VarAggr, StdAggr])
def test_stat_aggr(aggr):
    src = torch.randn(6, 64)
    index = torch.tensor([0, 1, 0, 1, 2, 1])
    out = aggr()(src, index)
    assert out.shape[0] == index.unique().shape[0]
