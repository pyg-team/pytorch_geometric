import pytest
import torch

from torch_geometric.nn import MedianAggregation, QuantileAggregation


@pytest.mark.parametrize('q', [0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
@pytest.mark.parametrize('interpolation', QuantileAggregation.interpolations)
@pytest.mark.parametrize('dim', [0, 1])
def test_quantile_aggregation(q, interpolation, dim):
    x = torch.tensor([
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
        [9.0, 0.0, 1.0],
        [2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0],
        [8.0, 9.0, 0.0],
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    index = torch.zeros(x.size(dim), dtype=torch.long)

    aggr = QuantileAggregation(q=q, interpolation=interpolation)
    assert str(aggr) == f"QuantileAggregation(q={q})"

    out = aggr(x, index, dim=dim)
    expected = x.quantile(q, dim, interpolation=interpolation, keepdim=True)
    assert torch.allclose(out, expected)


def test_median_aggregation():
    x = torch.tensor([
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
        [9.0, 0.0, 1.0],
        [2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0],
        [8.0, 9.0, 0.0],
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])

    aggr = MedianAggregation()
    assert str(aggr) == "MedianAggregation()"

    index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    assert aggr(x, index).tolist() == [
        [3.0, 1.0, 2.0],
        [5.0, 6.0, 4.0],
        [4.0, 5.0, 6.0],
    ]

    index = torch.tensor([0, 1, 0])
    assert aggr(x, index, dim=1).tolist() == [
        [0.0, 1.0],
        [3.0, 4.0],
        [6.0, 7.0],
        [1.0, 0.0],
        [2.0, 3.0],
        [5.0, 6.0],
        [0.0, 9.0],
        [1.0, 2.0],
        [4.0, 5.0],
        [7.0, 8.0],
    ]


def test_quantile_aggregation_multi():
    x = torch.tensor([
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
        [9.0, 0.0, 1.0],
        [2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0],
        [8.0, 9.0, 0.0],
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])

    qs = [0.25, 0.5, 0.75]

    assert torch.allclose(
        QuantileAggregation(qs)(x, index),
        torch.cat([QuantileAggregation(q)(x, index) for q in qs], dim=-1),
    )


def test_quantile_aggregation_validate():
    with pytest.raises(ValueError, match="at least one quantile"):
        QuantileAggregation(q=[])

    with pytest.raises(ValueError, match="must be in the range"):
        QuantileAggregation(q=-1)

    with pytest.raises(ValueError, match="Invalid interpolation method"):
        QuantileAggregation(q=0.5, interpolation=None)
