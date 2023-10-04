import torch

from torch_geometric.nn import SoftMedianAggregation, WeightedMedianAggregation


def test_weighted_median():
    aggr = WeightedMedianAggregation()
    assert str(aggr) == 'WeightedMedianAggregation()'

    # Single segment

    # Unweighted dimension-wise median
    x = torch.tensor([[1, 1, 5, 1], [1, 2, 3, 1], [1, 4, 4, 3], [1, 3, 2, 3],
                      [1, 5, 1, 2]]).float()
    index = torch.tensor([0, 0, 0, 0, 0])
    out = aggr(x, index)
    assert (out == torch.tensor([1, 3, 3, 2])).all()

    # Weighted dimension-wise median
    x = torch.arange(20).reshape(5, 4).float()
    # For simplicity the weights sum up to 1, otherwise the output is scaled
    weight = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4])
    index = torch.tensor([0, 0, 0, 0, 0])
    out = aggr(x, index, weight)
    assert (out == x[3, :]).all()

    # Single segment make weights sum to 2
    weight = 2 * torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4])
    out = aggr(x, index, weight)
    assert torch.allclose(out, 2 * x[3, :])

    # Single segment make weights sum to 1/2
    weight = 1 / 2 * torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4])
    out = aggr(x, index, weight)
    assert torch.allclose(out, 1 / 2 * x[3, :])

    # Two segments

    weight = torch.tensor([0.2, 0.4, 0.4, 0.6, 0.4])
    index = torch.tensor([0, 0, 0, 1, 1])
    out = aggr(x, index, weight)
    assert (out[0] == x[1, :]).all()
    assert (out[1] == x[3, :]).all()

    weight = torch.tensor([0.2, 0.4, 0.4, 2 * 0.6, 2 * 0.4])
    out = aggr(x, index, weight)
    assert torch.allclose(out[0], x[1, :])
    assert torch.allclose(out[1], 2 * x[3, :])

    weight = torch.tensor([0.2, 0.4, 0.4, 0.4, 0.6])
    out = aggr(x, index, weight)
    assert (out[0] == x[1, :]).all()
    assert (out[1] == x[4, :]).all()

    # Permute order (median is invariant w.r.t. input permutations)
    p = torch.tensor([3, 0, 4, 2, 1])

    weight = torch.tensor([0.2, 0.4, 0.4, 0.6, 0.4])
    index = torch.tensor([0, 0, 0, 1, 1])
    out = aggr(x[p], index[p], weight[p])
    assert (out[0] == x[1, :]).all()
    assert (out[1] == x[3, :]).all()


def test_soft_median():
    aggr = SoftMedianAggregation(T=1e-6)  # Small temperature to approx. hard
    assert str(aggr) == 'SoftMedianAggregation(T=1e-06, p=2.0)'

    # Single segment
    x = torch.arange(20).reshape(5, 4).float()
    # For simplicity the weights sum up to 1, otherwise the output is scaled
    weight = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4])
    index = torch.tensor([0, 0, 0, 0, 0])
    out = aggr(x, index, weight)
    assert (out == x[3, :]).all()

    # Single segment make weights sum to 2
    weight = 2 * torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4])
    out = aggr(x, index, weight)
    assert torch.allclose(out, 2 * x[3, :])

    # Single segment make weights sum to 1/2
    weight = 1 / 2 * torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4])
    out = aggr(x, index, weight)
    assert torch.allclose(out, 1 / 2 * x[3, :])

    # Two segments
    weight = torch.tensor([0.2, 0.4, 0.4, 0.6, 0.4])
    index = torch.tensor([0, 0, 0, 1, 1])
    out = aggr(x, index, weight)
    assert torch.allclose(out[0], x[1, :])
    assert torch.allclose(out[1], x[3, :])

    weight = torch.tensor([0.2, 0.4, 0.4, 2 * 0.6, 2 * 0.4])
    out = aggr(x, index, weight)
    assert torch.allclose(out[0], x[1, :])
    assert torch.allclose(out[1], 2 * x[3, :])

    weight = torch.tensor([0.2, 0.4, 0.4, 0.4, 0.6])
    out = aggr(x, index, weight)
    assert torch.allclose(out[0], x[1, :])
    assert torch.allclose(out[1], x[4, :])

    # Test high temperature to make sure it mimics a weighted mean
    aggr = SoftMedianAggregation(T=1e6)

    out = aggr(x, index, weight)
    assert torch.allclose(out[0], (weight[:3, None] * x[:3, :]).sum(0))
    assert torch.allclose(out[1], (weight[3:, None] * x[3:, :]).sum(0))
