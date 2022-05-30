import pytest
import torch

from torch_geometric.nn import (
    MaxAggregation,
    MeanAggregation,
    MinAggregation,
    PowerMeanAggregation,
    SoftmaxAggregation,
    StdAggregation,
    SumAggregation,
    VarAggregation,
)


def test_validate():
    x = torch.randn(6, 16)
    index = torch.tensor([0, 0, 1, 1, 1, 2])
    ptr = torch.tensor([0, 2, 5, 6])

    aggr = MeanAggregation()

    with pytest.raises(ValueError, match="either 'index' or 'ptr'"):
        aggr(x)

    with pytest.raises(ValueError, match="invalid dimension"):
        aggr(x, index, dim=-3)

    with pytest.raises(ValueError, match="mismatch between"):
        aggr(x, ptr=ptr, dim_size=2)


@pytest.mark.parametrize('Aggregation', [
    MeanAggregation, SumAggregation, MaxAggregation, MinAggregation,
    VarAggregation, StdAggregation
])
def test_basic_aggregation(Aggregation):
    x = torch.randn(6, 16)
    index = torch.tensor([0, 0, 1, 1, 1, 2])
    ptr = torch.tensor([0, 2, 5, 6])

    aggr = Aggregation()
    assert str(aggr) == f'{Aggregation.__name__}()'

    out = aggr(x, index)
    assert out.size() == (3, x.size(1))
    assert torch.allclose(out, aggr(x, ptr=ptr))


@pytest.mark.parametrize('Aggregation',
                         [SoftmaxAggregation, PowerMeanAggregation])
@pytest.mark.parametrize('learn', [True, False])
def test_gen_aggregation(Aggregation, learn):
    x = torch.randn(6, 16)
    index = torch.tensor([0, 0, 1, 1, 1, 2])
    ptr = torch.tensor([0, 2, 5, 6])

    aggr = Aggregation(learn=learn)
    assert str(aggr) == f'{Aggregation.__name__}()'

    out = aggr(x, index)
    assert out.size() == (3, x.size(1))
    assert torch.allclose(out, aggr(x, ptr=ptr))

    if learn:
        out.mean().backward()
        for param in aggr.parameters():
            assert not torch.isnan(param.grad).any()
