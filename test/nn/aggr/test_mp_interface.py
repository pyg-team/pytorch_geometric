import pytest
import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_sparse.matmul import spmm

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

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)


@pytest.mark.parametrize('aggr_tuple', [(MeanAggregation, 'mean'),
                                        (SumAggregation, 'sum'),
                                        (MaxAggregation, 'max'),
                                        (MinAggregation, 'min'),
                                        (MulAggregation, 'mul'),
                                        (VarAggregation, 'var'),
                                        (StdAggregation, 'std')])
def test_basic_aggr_conv(aggr_tuple):
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))

    aggr_module, aggr_str = aggr_tuple
    conv1 = MyConv(aggr=aggr_module())
    out1 = conv1(x, edge_index)
    assert out1.size() == (4, 16)
    conv2 = MyConv(aggr=aggr_str)
    out2 = conv2(x, edge_index)
    assert out2.size() == (4, 16)
    assert torch.allclose(out1, out2)

    assert conv1(x, adj.t()).tolist() == out1.tolist()
    conv1.fuse = False
    assert conv1(x, adj.t()).tolist() == out1.tolist()

    assert conv2(x, adj.t()).tolist() == out2.tolist()
    conv2.fuse = False
    assert conv2(x, adj.t()).tolist() == out2.tolist()


@pytest.mark.parametrize('aggr_module',
                         [SoftmaxAggregation, PowerMeanAggregation])
@pytest.mark.parametrize('learn', [True, False])
def test_gen_aggr_conv(aggr_module, learn):
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    conv = MyConv(aggr=aggr_module(learn=learn))
    out = conv(x, edge_index)
    assert out.size() == (4, 16)


def test_lstm_aggr_conv():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    conv = MyConv(aggr=LSTMAggregation(16, 32))
    out = conv(x, edge_index)
    assert out.size() == (4, 32)


def test_multiple_aggr_conv():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    aggrs1 = ['mean', 'sum', 'max']
    conv1 = MyConv(aggr=MultiAggregation(aggrs=aggrs1))
    out1 = conv1(x, edge_index)
    assert out1.size() == (4, 48)
    assert not torch.allclose(out1[:, 0:16], out1[:, 16:32])
    assert not torch.allclose(out1[:, 0:16], out1[:, 32:48])
    assert not torch.allclose(out1[:, 16:32], out1[:, 32:48])

    aggrs2 = [MeanAggregation(), SumAggregation(), MaxAggregation()]
    conv2 = MyConv(aggr=aggrs2)
    out2 = conv2(x, edge_index)
    assert out2.size() == (4, 48)
    assert not torch.allclose(out2[:, 0:16], out2[:, 16:32])
    assert not torch.allclose(out2[:, 0:16], out2[:, 32:48])
    assert not torch.allclose(out2[:, 16:32], out2[:, 32:48])

    assert torch.allclose(out1, out2)
