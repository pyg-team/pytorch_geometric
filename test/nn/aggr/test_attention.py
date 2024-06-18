import pytest
import torch

import torch_geometric.typing
from torch_geometric.nn import MLP
from torch_geometric.nn.aggr import AttentionalAggregation


@pytest.mark.parametrize('dim', [2, 3])
def test_attentional_aggregation(dim):
    channels = 16
    x = torch.randn(6, channels) if dim == 2 else torch.randn(2, 6, channels)
    index = torch.tensor([0, 0, 1, 1, 1, 2])
    ptr = torch.tensor([0, 2, 5, 6])

    gate_nn = MLP([channels, 1], act='relu')
    nn = MLP([channels, channels], act='relu')
    aggr = AttentionalAggregation(gate_nn, nn)
    aggr.reset_parameters()
    assert str(aggr) == (f'AttentionalAggregation(gate_nn=MLP({channels}, 1), '
                         f'nn=MLP({channels}, {channels}))')

    out = aggr(x, index)
    assert out.size() == (3, channels) if dim == 2 else (2, 3, channels)

    if (not torch_geometric.typing.WITH_TORCH_SCATTER
            and (dim == 3 or not torch_geometric.typing.WITH_PT20)):
        with pytest.raises(ImportError, match="requires the 'torch-scatter'"):
            aggr(x, ptr=ptr)
    else:
        assert torch.allclose(out, aggr(x, ptr=ptr))
