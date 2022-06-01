from itertools import combinations

import pytest
import torch
from torch import Tensor

from torch_geometric.nn import (
    LSTMAggregation,
    MaxAggregation,
    MeanAggregation,
    MessagePassing,
    MinAggregation,
    MulAggregation,
    MultiAggregation,
    PowerMeanAggregation,
    SoftmaxAggregation,
    StdAggregation,
    SumAggregation,
    VarAggregation,
)
from torch_geometric.typing import Adj


class MyConv(MessagePassing):
    def __init__(self, aggr='mean'):
        super().__init__(aggr=aggr)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        # propagate_type: (x: Tensor)
        return self.propagate(edge_index, x=x, size=None)


@pytest.mark.parametrize(
    'aggr', [(MeanAggregation, 'mean'), (SumAggregation, 'sum'),
             (MaxAggregation, 'max'), (MinAggregation, 'min'),
             (MulAggregation, 'mul'), VarAggregation, StdAggregation])
def test_my_basic_aggr_conv(aggr):
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    if isinstance(aggr, tuple):
        aggr_module, aggr_str = aggr
        conv1 = MyConv(aggr=aggr_module())
        out1 = conv1(x, edge_index)
        assert out1.size() == (4, 16)

        conv2 = MyConv(aggr=aggr_str)
        out2 = conv2(x, edge_index)
        assert out2.size() == (4, 16)

        assert torch.allclose(out1, out2)
    else:
        conv = MyConv(aggr=aggr())
        out = conv(x, edge_index)
        assert out.size() == (4, 16)


@pytest.mark.parametrize('Aggregation',
                         [SoftmaxAggregation, PowerMeanAggregation])
@pytest.mark.parametrize('learn', [True, False])
def test_my_gen_aggr_conv(Aggregation, learn):
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    conv = MyConv(aggr=Aggregation(learn=learn))
    out = conv(x, edge_index)
    assert out.size() == (4, 16)


def test_my_lstm_aggr_conv():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    conv = MyConv(aggr=LSTMAggregation(16, 32))
    out = conv(x, edge_index)
    assert out.size() == (4, 32)


aggr_list = [
    'mean', 'sum', 'max', 'min', 'mul', 'std', 'var',
    SoftmaxAggregation(learn=True),
    PowerMeanAggregation(learn=True),
    LSTMAggregation(16, 16)
]
aggrs = [list(aggr) for aggr in combinations(aggr_list, 3)]


@pytest.mark.parametrize('aggrs', aggrs)
def test_my_multiple_aggr_conv(aggrs):
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = MyConv(aggr=MultiAggregation(aggrs=aggrs))
    out = conv(x, edge_index)
    assert out.size() == (4, 48)
    assert not torch.allclose(out[:, 0:16], out[:, 16:32])
    assert not torch.allclose(out[:, 0:16], out[:, 32:48])
    assert not torch.allclose(out[:, 16:32], out[:, 32:48])
