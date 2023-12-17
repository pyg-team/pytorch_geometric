import pytest
import torch

from torch_geometric.nn import MixerAggregation


def test_basic():
    x = torch.randn(6, 16)
    index = torch.tensor([0, 0, 1, 1, 1, 2])

    aggr = MixerAggregation(16, 32)
    assert str(aggr) == 'MixerAggregation(16, 32)'

    with pytest.raises(ValueError, match="is not sorted"):
        aggr(x, torch.tensor([0, 1, 0, 1, 2, 1]))

    out = aggr(x, index)
    assert out.size() == (3, 32)
