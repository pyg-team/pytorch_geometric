import pytest
import torch

from torch_geometric.nn import LCMAggregation


def test_lcm_aggregation_with_project():
    x = torch.randn(6, 16)
    index = torch.tensor([0, 0, 1, 1, 1, 2])

    aggr = LCMAggregation(16, 32)
    assert str(aggr) == 'LCMAggregation(16, 32, project=True)'

    out = aggr(x, index)
    assert out.size() == (3, 32)


def test_lcm_aggregation_without_project():
    x = torch.randn(6, 16)
    index = torch.tensor([0, 0, 1, 1, 1, 2])

    aggr = LCMAggregation(16, 16, project=False)
    assert str(aggr) == 'LCMAggregation(16, 16, project=False)'

    out = aggr(x, index)
    assert out.size() == (3, 16)


def test_lcm_aggregation_error_handling():
    with pytest.raises(ValueError, match="must be projected"):
        LCMAggregation(16, 32, project=False)
