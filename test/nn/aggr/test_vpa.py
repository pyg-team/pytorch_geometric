import pytest
import torch

from torch_geometric.nn import (
    MeanAggregation,
    SumAggregation,
    VariancePreservingAggregation,
)


def test_vpa():
    x = torch.randn(6, 16)
    index = torch.tensor([0, 0, 1, 1, 1, 2])
    ptr = torch.tensor([0, 2, 5, 6])

    vpa_aggr = VariancePreservingAggregation()
    mean_aggr = MeanAggregation()
    sum_aggr = SumAggregation()

    out_vpa = vpa_aggr(x, index)
    out_mean = mean_aggr(x, index)
    out_sum = sum_aggr(x, index)

    # equivalent formulation
    out_vpa2 = torch.sqrt(out_mean.abs() * out_sum.abs()) * torch.sign(out_sum)

    assert out_vpa.size() == (3, x.size(1))
    assert torch.allclose(out_vpa, out_vpa2)
    assert torch.allclose(out_vpa, vpa_aggr(x, ptr=ptr))
