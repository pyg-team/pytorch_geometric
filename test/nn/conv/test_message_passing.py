import copy
from typing import Tuple, Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size

import pytest
import torch
from torch import Tensor
from torch.nn import Linear
from torch_sparse import SparseTensor
from torch_sparse.matmul import spmm
from torch_geometric.nn import MessagePassing


class MyConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int):
        """"""
        super(MyConv, self).__init__(aggr='add')

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels)
        self.lin_r = Linear(in_channels[1], out_channels)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        return spmm(adj_t, x[0], reduce=self.aggr)


def test_my_conv():
    x1 = torch.randn(4, 8)
    x2 = torch.randn(2, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    value = torch.randn(row.size(0))
    adj = SparseTensor(row=row, col=col, value=value, sparse_sizes=(4, 4))

    conv = MyConv(8, 32)
    out = conv(x1, edge_index, value)
    assert out.size() == (4, 32)
    assert conv(x1, edge_index, value, (4, 4)).tolist() == out.tolist()
    assert conv(x1, adj.t()).tolist() == out.tolist()
    conv.fuse = False
    assert conv(x1, adj.t()).tolist() == out.tolist()
    conv.fuse = True

    t = '(Tensor, Tensor, OptTensor, Size) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert jit(x1, edge_index, value).tolist() == out.tolist()
    assert jit(x1, edge_index, value, (4, 4)).tolist() == out.tolist()

    t = '(Tensor, SparseTensor, OptTensor, Size) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert jit(x1, adj.t()).tolist() == out.tolist()
    jit.fuse = False
    assert jit(x1, adj.t()).tolist() == out.tolist()
    jit.fuse = True

    adj = adj.sparse_resize((4, 2))
    conv = MyConv((8, 16), 32)
    out1 = conv((x1, x2), edge_index, value)
    out2 = conv((x1, None), edge_index, value, (4, 2))
    assert out1.size() == (2, 32)
    assert out2.size() == (2, 32)
    assert conv((x1, x2), edge_index, value, (4, 2)).tolist() == out1.tolist()
    assert conv((x1, x2), adj.t()).tolist() == out1.tolist()
    assert conv((x1, None), adj.t()).tolist() == out2.tolist()
    conv.fuse = False
    assert conv((x1, x2), adj.t()).tolist() == out1.tolist()
    assert conv((x1, None), adj.t()).tolist() == out2.tolist()
    conv.fuse = True

    t = '(OptPairTensor, Tensor, OptTensor, Size) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert jit((x1, x2), edge_index, value).tolist() == out1.tolist()
    assert jit((x1, x2), edge_index, value, (4, 2)).tolist() == out1.tolist()
    assert jit((x1, None), edge_index, value, (4, 2)).tolist() == out2.tolist()

    t = '(OptPairTensor, SparseTensor, OptTensor, Size) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert jit((x1, x2), adj.t()).tolist() == out1.tolist()
    assert jit((x1, None), adj.t()).tolist() == out2.tolist()
    jit.fuse = False
    assert jit((x1, x2), adj.t()).tolist() == out1.tolist()
    assert jit((x1, None), adj.t()).tolist() == out2.tolist()
    jit.fuse = True


def test_my_static_graph_conv():
    x1 = torch.randn(3, 4, 8)
    x2 = torch.randn(3, 2, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    value = torch.randn(row.size(0))
    adj = SparseTensor(row=row, col=col, value=value, sparse_sizes=(4, 4))

    conv = MyConv(8, 32)
    out = conv(x1, edge_index, value)
    assert out.size() == (3, 4, 32)
    assert conv(x1, edge_index, value, (4, 4)).tolist() == out.tolist()
    assert conv(x1, adj.t()).tolist() == out.tolist()

    adj = adj.sparse_resize((4, 2))
    conv = MyConv((8, 16), 32)
    out1 = conv((x1, x2), edge_index, value)
    assert out1.size() == (3, 2, 32)
    assert conv((x1, x2), edge_index, value, (4, 2)).tolist() == out1.tolist()
    assert conv((x1, x2), adj.t()).tolist() == out1.tolist()
    out2 = conv((x1, None), edge_index, value, (4, 2))
    assert out2.size() == (3, 2, 32)
    assert conv((x1, None), adj.t()).tolist() == out2.tolist()


def test_copy():
    conv = MyConv(8, 32)
    conv2 = copy.copy(conv)

    assert conv != conv2
    assert conv.lin_l.weight.tolist() == conv2.lin_l.weight.tolist()
    assert conv.lin_r.weight.tolist() == conv2.lin_r.weight.tolist()
    assert conv.lin_l.weight.data_ptr == conv2.lin_l.weight.data_ptr
    assert conv.lin_r.weight.data_ptr == conv2.lin_r.weight.data_ptr

    conv = copy.deepcopy(conv)
    assert conv != conv2
    assert conv.lin_l.weight.tolist() == conv2.lin_l.weight.tolist()
    assert conv.lin_r.weight.tolist() == conv2.lin_r.weight.tolist()
    assert conv.lin_l.weight.data_ptr != conv2.lin_l.weight.data_ptr
    assert conv.lin_r.weight.data_ptr != conv2.lin_r.weight.data_ptr


class MyDefaultArgConv(MessagePassing):
    def __init__(self):
        super(MyDefaultArgConv, self).__init__(aggr='mean')

    # propagate_type: (x: Tensor)
    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_j, zeros: bool = True):
        return x_j * 0 if zeros else x_j


def test_my_default_arg_conv():
    x = torch.randn(4, 1)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))

    conv = MyDefaultArgConv()
    assert conv(x, edge_index).view(-1).tolist() == [0, 0, 0, 0]
    assert conv(x, adj.t()).view(-1).tolist() == [0, 0, 0, 0]

    # This should not succeed in JIT mode.
    with pytest.raises(RuntimeError):
        torch.jit.script(conv.jittable())
