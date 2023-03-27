import pytest
import torch

import torch_geometric.typing
from torch_geometric.nn import MLP
from torch_geometric.nn.aggr import AttentionalAggregation


def test_attentional_aggregation():
    channels = 16
    x = torch.randn(6, channels)
    index = torch.tensor([0, 0, 1, 1, 1, 2])
    ptr = torch.tensor([0, 2, 5, 6])

    gate_nn = MLP([channels, 1], act='relu')
    nn = MLP([channels, channels], act='relu')
    aggr = AttentionalAggregation(gate_nn, nn)
    assert str(aggr) == (f'AttentionalAggregation(gate_nn=MLP({channels}, 1), '
                         f'nn=MLP({channels}, {channels}))')

    out = aggr(x, index)
    assert out.size() == (3, channels)

    if not torch_geometric.typing.WITH_TORCH_SCATTER:
        with pytest.raises(ImportError, match="'segment' requires"):
            aggr(x, ptr=ptr)
    else:
        assert torch.allclose(out, aggr(x, ptr=ptr))
