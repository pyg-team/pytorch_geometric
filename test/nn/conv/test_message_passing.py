import copy
import pytest
from typing import Tuple, Optional

import torch
from torch_sparse import SparseTensor
from torch_sparse.matmul import spmm
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

edge_index = torch.tensor([
    [0, 0, 0, 1, 1],
    [0, 1, 2, 0, 2],
])
adj_t = SparseTensor(row=edge_index[1], col=edge_index[0])
x = (
    torch.arange(1, 3, dtype=torch.float),
    torch.arange(1, 4, dtype=torch.float),
)


class MyBasicConv(MessagePassing):
    def __init__(self):
        super(MyBasicConv, self).__init__()

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)


def test_my_basic_conv():
    conv = MyBasicConv()
    out = conv(x[1], edge_index)
    assert out.tolist() == [3.0, 1.0, 3.0]
    assert conv(x, edge_index).tolist() == out.tolist()
    assert conv(x[0], adj_t).tolist() == out.tolist()
    assert conv(x, adj_t).tolist() == out.tolist()

    jitted = conv.jittable(x=x[1], edge_index=edge_index)
    jitted = torch.jit.script(jitted)
    assert jitted(x[1], edge_index).tolist() == out.tolist()

    with pytest.raises(RuntimeError):
        jitted = conv.jittable(x=x, edge_index=edge_index)
        jitted = torch.jit.script(jitted)
        jitted = conv.jittable(x=x[0], edge_index=adj_t)
        jitted = torch.jit.script(jitted)
        jitted = conv.jittable(x=x, edge_index=adj_t)
        jitted = torch.jit.script(jitted)


class MyAdvancedConv(MessagePassing):
    def __init__(self):
        super(MyAdvancedConv, self).__init__(aggr='add')

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j, edge_index_i, size_i: Optional[int]):
        assert size_i is not None
        alpha = softmax(x_j, edge_index_i, num_nodes=size_i)
        return alpha * x_j

    def update(self, inputs):
        return inputs + 2


def test_my_advanced_conv():
    conv = MyAdvancedConv()
    out = conv(x[1], edge_index)
    assert conv(x, edge_index).tolist() == out.tolist()
    assert conv(x[0], adj_t).tolist() == out.tolist()
    assert conv(x, adj_t).tolist() == out.tolist()

    jitted = conv.jittable(x=x[1], edge_index=edge_index)
    jitted = torch.jit.script(jitted)
    assert jitted(x[1], edge_index).tolist() == out.tolist()

    with pytest.raises(RuntimeError):
        jitted = conv.jittable(x=x, edge_index=edge_index)
        jitted = torch.jit.script(jitted)
        jitted = conv.jittable(x=x[0], edge_index=adj_t)
        jitted = torch.jit.script(jitted)
        jitted = conv.jittable(x=x, edge_index=adj_t)
        jitted = torch.jit.script(jitted)


class MyDefaultArgumentsConv(MessagePassing):
    def __init__(self):
        super(MyDefaultArgumentsConv, self).__init__(aggr='mean')

    def forward(self, x, edge_index, **kwargs):
        return self.propagate(edge_index, x=x, **kwargs)

    def message(self, x_j, zeros: bool = True):
        return x_j * 0 if zeros else x_j


def test_my_default_arguments_conv():
    conv = MyDefaultArgumentsConv()
    out = conv(x[1], edge_index)
    assert out.tolist() == [0, 0, 0]
    assert conv(x, edge_index).tolist() == out.tolist()
    assert conv(x[0], adj_t).tolist() == out.tolist()
    assert conv(x, adj_t).tolist() == out.tolist()

    with pytest.raises(torch.jit.frontend.NotSupportedError):
        jitted = conv.jittable(x=x[1], edge_index=edge_index)
        jitted = torch.jit.script(jitted)
        jitted = conv.jittable(x=x, edge_index=edge_index)
        jitted = torch.jit.script(jitted)
        jitted = conv.jittable(x=x[0], edge_index=adj_t)
        jitted = torch.jit.script(jitted)
        jitted = conv.jittable(x=x, edge_index=adj_t)
        jitted = torch.jit.script(jitted)


class MyBipartiteConv(MessagePassing):
    def __init__(self):
        super(MyBipartiteConv, self).__init__(aggr='mean')

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], edge_index,
                size: Optional[Tuple[int, int]] = None):
        return self.propagate(edge_index, size=size, x=x)

    def update(self, inputs, x: Tuple[torch.Tensor, torch.Tensor]):
        if isinstance(x, tuple):
            x = x[-1]
        return inputs + x


def test_my_bipartite_conv():
    conv = MyBipartiteConv()
    out = conv(x[1], edge_index)
    assert out.tolist() == [2.5, 3.0, 4.5]
    assert conv(x, edge_index).tolist() == out.tolist()
    assert conv(x, adj_t).tolist() == out.tolist()

    with pytest.raises(RuntimeError):
        conv(x[0], adj_t)

    jitted = conv.jittable(x=x, edge_index=edge_index)
    jitted = torch.jit.script(jitted)
    assert jitted(x, edge_index).tolist() == out.tolist()

    with pytest.raises(RuntimeError):
        jitted = conv.jittable(x=x[1], edge_index=edge_index)
        jitted = torch.jit.script(jitted)
        jitted = conv.jittable(x=x[0], edge_index=adj_t)
        jitted = torch.jit.script(jitted)
        jitted = conv.jittable(x=x, edge_index=adj_t)
        jitted = torch.jit.script(jitted)


class MyDoublePropagateConv(MessagePassing):
    def __init__(self):
        super(MyDoublePropagateConv, self).__init__(aggr='max')

    def forward(self, x, edge_index):
        self.flow = 'source_to_target'
        out1 = self.propagate(edge_index, x=x)
        self.flow = 'target_to_source'
        out2 = self.propagate(edge_index, x=x)
        return out1 + out2


def test_my_double_propagate_conv():
    conv = MyDoublePropagateConv()
    out = conv(x[1], edge_index)
    assert out.tolist() == [5.0, 4.0, 2.0]

    with pytest.raises(ValueError):
        assert conv(x[0], adj_t).tolist() == out.tolist()

    jitted = conv.jittable(x=x[1], edge_index=edge_index)
    jitted = torch.jit.script(jitted)
    assert jitted(x[1], edge_index).tolist() == out.tolist()


class MyMessageAndAggregateConv1(MessagePassing):
    def __init__(self):
        super(MyMessageAndAggregateConv1, self).__init__()

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message_and_aggregate(self, adj_t: SparseTensor, x):
        assert adj_t is not None
        return spmm(adj_t, x.view(-1, 1)).view(-1)


class MyMessageAndAggregateConv2(MessagePassing):
    def __init__(self):
        super(MyMessageAndAggregateConv2, self).__init__()

    def forward(self, x, edge_index: SparseTensor):
        return self.propagate(edge_index, x=x)

    def message_and_aggregate(self, adj_t: SparseTensor, x):
        assert adj_t is not None
        return spmm(adj_t, x.view(-1, 1)).view(-1)


def test_my_message_and_aggregate_conv():
    conv = MyMessageAndAggregateConv1()

    out = conv(x[1], edge_index)
    assert out.tolist() == [3.0, 1.0, 3.0]
    assert conv(x[0], adj_t).tolist() == out.tolist()

    jitted = conv.jittable(x=x[1], edge_index=edge_index)
    jitted = torch.jit.script(jitted)
    assert jitted(x[1], edge_index).tolist() == [3.0, 1.0, 3.0]

    conv = MyMessageAndAggregateConv2()

    out = conv(x[1], edge_index)
    assert out.tolist() == [3.0, 1.0, 3.0]
    assert conv(x[0], adj_t).tolist() == out.tolist()

    jitted = conv.jittable(x=x[0], edge_index=adj_t)
    jitted = torch.jit.script(jitted)
    assert jitted(x[0], adj_t).tolist() == [3.0, 1.0, 3.0]


class MyCopyConv(MessagePassing):
    def __init__(self):
        super(MyCopyConv, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(10))

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)


def test_copy():
    conv = MyCopyConv()
    conv2 = copy.copy(conv)

    assert conv != conv2
    assert conv.weight.data_ptr == conv2.weight.data_ptr

    conv = copy.deepcopy(conv)
    assert conv != conv2
    assert conv.weight.data_ptr != conv2.weight.data_ptr
