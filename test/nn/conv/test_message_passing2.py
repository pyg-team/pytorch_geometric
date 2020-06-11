import copy
from typing import Optional
from torch_geometric.typing import OptPairTensor, Size

import pytest
import torch
from torch import Tensor
from torch.nn import Linear
from torch_sparse import SparseTensor
from torch_sparse.matmul import spmm
from torch_geometric.nn import MessagePassing


class MyConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MyConv, self).__init__(aggr='add')
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.lin_rel = Linear(in_channels[0], out_channels)
        self.lin_root = Linear(in_channels[1], out_channels)

    # propagate_type: (x: OptPairTensor, edge_weight: Optional[Tensor])
    def forward(self, x, edge_index, edge_weight=None, size=None):
        # type: (Tensor, Tensor, Optional[Tensor], Size) -> Tensor
        # type: (OptPairTensor, Tensor, Optional[Tensor], Size) -> Tensor
        # type: (Tensor, SparseTensor, Optional[Tensor], Size) -> Tensor
        # type: (OptPairTensor, SparseTensor, Optional[Tensor], Size) -> Tensor
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size)
        out = self.lin_rel(out)
        x_root = x[1]
        if x_root is not None:
            out += self.lin_root(x_root)
        return out

    def message(self, x_j: Tensor, edge_weight: Optional[Tensor]) -> Tensor:
        return x_j if edge_weight is None else x_j * edge_weight.view(-1, 1)

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        return spmm(adj_t, x[0], reduce=self.aggr)


class HomoEdgeIndexNet(torch.nn.Module):
    def __init__(self):
        super(HomoEdgeIndexNet, self).__init__()
        torch.manual_seed(42)
        self.conv = MyConv(8, 32)

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_weight: Optional[Tensor] = None,
                size: Size = None) -> Tensor:
        return self.conv(x, edge_index, edge_weight, size)


class HomoSparseTensorNet(torch.nn.Module):
    def __init__(self):
        super(HomoSparseTensorNet, self).__init__()
        torch.manual_seed(42)
        self.conv = MyConv(8, 32)

    def forward(self, x: Tensor, adj_t: SparseTensor) -> Tensor:
        return self.conv(x, adj_t)


class HeteEdgeIndexNet(torch.nn.Module):
    def __init__(self):
        super(HeteEdgeIndexNet, self).__init__()
        torch.manual_seed(42)
        self.conv = MyConv((8, 16), 32)

    def forward(self, x: OptPairTensor, edge_index: Tensor,
                edge_weight: Optional[Tensor] = None,
                size: Size = None) -> Tensor:
        return self.conv(x, edge_index, edge_weight, size)


class HeteSparseTensorNet(torch.nn.Module):
    def __init__(self):
        super(HeteSparseTensorNet, self).__init__()
        torch.manual_seed(42)
        self.conv = MyConv((8, 16), 32)

    def forward(self, x: OptPairTensor, adj_t: SparseTensor) -> Tensor:
        return self.conv(x, adj_t)


def test_my_conv():
    x1 = torch.randn(4, 8)
    x2 = torch.randn(2, 16)
    edge_index = torch.tensor([
        [0, 1, 2, 3],
        [0, 0, 1, 1],
    ])
    row, col = edge_index
    assert row.max() < x1.size(0)
    assert col.max() < x2.size(0)
    edge_weight = torch.randn(row.size(0))
    adj = SparseTensor(row=row, col=col, value=edge_weight,
                       sparse_sizes=(4, 4))

    homo_edge_index = HomoEdgeIndexNet()
    homo_sparse_tensor = HomoSparseTensorNet()

    out1 = homo_edge_index(x1, edge_index, edge_weight)
    out2 = homo_edge_index(x1, edge_index, edge_weight, (4, 4))
    out3 = homo_sparse_tensor(x1, adj.t())
    assert out1.size() == (4, 32)
    assert out1.tolist() == out2.tolist()
    assert out1.tolist() == out3.tolist()

    adj = adj.sparse_resize((4, 2))

    hete_edge_index = HeteEdgeIndexNet()
    hete_sparse_tensor = HeteSparseTensorNet()

    out1 = hete_edge_index((x1, x2), edge_index, edge_weight)
    out2 = hete_edge_index((x1, x2), edge_index, edge_weight, (4, 2))
    out3 = hete_sparse_tensor((x1, x2), adj.t())
    assert out1.size() == (2, 32)
    assert out1.tolist() == out2.tolist()
    assert out1.tolist() == out3.tolist()

    out1 = hete_edge_index((x1, None), edge_index, edge_weight, (4, 2))
    out2 = hete_sparse_tensor((x1, None), adj.t())
    assert out1.size() == (2, 32)
    assert out1.tolist() == out2.tolist()

    out = hete_edge_index((x1, None), edge_index, edge_weight)
    assert out.size() == (4, 32)

    adj = adj.sparse_resize((4, 4))

    homo_edge_index.conv = homo_edge_index.conv.jittable()
    homo_edge_index = torch.jit.script(homo_edge_index)
    homo_sparse_tensor.conv = homo_sparse_tensor.conv.jittable()
    homo_sparse_tensor = torch.jit.script(homo_sparse_tensor)

    out1 = homo_edge_index(x1, edge_index, edge_weight)
    out2 = homo_edge_index(x1, edge_index, edge_weight, (4, 4))
    out3 = homo_sparse_tensor(x1, adj.t())
    assert out1.size() == (4, 32)
    assert out1.tolist() == out2.tolist()
    assert out1.tolist() == out3.tolist()

    adj = adj.sparse_resize((4, 2))

    hete_edge_index.conv = hete_edge_index.conv.jittable()
    hete_edge_index = torch.jit.script(hete_edge_index)
    hete_sparse_tensor.conv = hete_sparse_tensor.conv.jittable()
    hete_sparse_tensor = torch.jit.script(hete_sparse_tensor)

    out1 = hete_edge_index((x1, x2), edge_index, edge_weight)
    out2 = hete_edge_index((x1, x2), edge_index, edge_weight, (4, 2))
    out3 = hete_sparse_tensor((x1, x2), adj.t())
    assert out1.size() == (2, 32)
    assert out1.tolist() == out2.tolist()
    assert out1.tolist() == out3.tolist()

    out1 = hete_edge_index((x1, None), edge_index, edge_weight, (4, 2))
    out2 = hete_sparse_tensor((x1, None), adj.t())
    assert out1.size() == (2, 32)
    assert out1.tolist() == out2.tolist()

    out = hete_edge_index((x1, None), edge_index, edge_weight)
    assert out.size() == (4, 32)


def test_copy():
    conv = MyConv(8, 32)
    conv2 = copy.copy(conv)

    assert conv != conv2
    assert conv.lin_rel.weight.tolist() == conv2.lin_rel.weight.tolist()
    assert conv.lin_root.weight.tolist() == conv2.lin_root.weight.tolist()
    assert conv.lin_rel.weight.data_ptr == conv2.lin_rel.weight.data_ptr
    assert conv.lin_root.weight.data_ptr == conv2.lin_root.weight.data_ptr

    conv = copy.deepcopy(conv)
    assert conv != conv2
    assert conv.lin_rel.weight.tolist() == conv2.lin_rel.weight.tolist()
    assert conv.lin_root.weight.tolist() == conv2.lin_root.weight.tolist()
    assert conv.lin_rel.weight.data_ptr != conv2.lin_rel.weight.data_ptr
    assert conv.lin_root.weight.data_ptr != conv2.lin_root.weight.data_ptr


class MyDefaultArgConv(MessagePassing):
    def __init__(self):
        super(MyDefaultArgConv, self).__init__(aggr='mean')

    # propagate_type: (x: Tensor)
    def forward(self, x, edge_index):
        # type: (Tensor, Tensor) -> Tensor
        # type: (Tensor, SparseTensor) -> Tensor
        return self.propagate(edge_index, x=x)

    def message(self, x_j, zeros: bool = True):
        return x_j * 0 if zeros else x_j


def test_my_default_arg_conv():
    x = torch.randn(4)
    edge_index = torch.tensor([
        [0, 1, 2, 3],
        [0, 0, 1, 1],
    ])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))

    conv = MyDefaultArgConv()
    assert conv(x, edge_index).tolist() == [0, 0, 0, 0]
    assert conv(x, adj.t()).tolist() == [0, 0, 0, 0]

    # This should not succeed in JIT mode.
    with pytest.raises(RuntimeError):
        torch.jit.script(conv.jittable())
