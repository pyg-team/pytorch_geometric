import pytest
import torch

from torch_geometric.nn import MedianAggregation, QuantileAggregation

x = torch.tensor([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [9, 0, 1],
    [2, 3, 4],
    [5, 6, 7],
    [8, 9, 0],
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]).float()


@pytest.mark.parametrize('index,result,dim', [(
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2]),
        torch.tensor([
            [3, 1, 2],
            [5, 6, 4],
            [4, 5, 6],
        ]).float(),
        0,
    ), (
        torch.tensor([0, 1, 0]),
        torch.cat([x[:, ::2].min(-1, keepdim=True)[0], x[:, [1]]], dim=1),
        1
    ), (
        torch.zeros_like(x[:, 0]).long(),
        torch.median(x, 0, keepdim=True)[0],
        0,
    ), (
        torch.zeros_like(x[0, :]).long(),
        torch.median(x, 1, keepdim=True)[0],
        1,
)])
def test_median_aggregation(index, dim, result):
    aggr = MedianAggregation()
    assert torch.allclose(result, aggr(x=x, index=index, dim=dim))


@pytest.mark.parametrize('q',
                         [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
@pytest.mark.parametrize('interpolation', QuantileAggregation.interpolations)
@pytest.mark.parametrize('dim', [0, 1])
def test_quantile_aggregation(q, interpolation, dim):
    aggr = QuantileAggregation(q=q, interpolation=interpolation)
    exp = torch.quantile(x, q=q, interpolation=interpolation, dim=dim,
                         keepdims=True)
    obs = aggr(x, torch.zeros(x.size(dim), dtype=torch.long, device=x.device),
               dim=dim)

    assert torch.allclose(exp, obs)
