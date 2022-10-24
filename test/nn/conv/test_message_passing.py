import copy
from typing import Tuple, Union

import pytest
import torch
from torch import Tensor
from torch.nn import Linear
from torch_scatter import scatter
from torch_sparse import SparseTensor
from torch_sparse.matmul import spmm

from torch_geometric.nn import MessagePassing, aggr
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size


class MyConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, aggr: str = 'add'):
        super().__init__(aggr=aggr)

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels)
        self.lin_r = Linear(in_channels[1], out_channels)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:
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


def test_my_conv_out_of_bounds():
    x = torch.randn(3, 8)
    value = torch.randn(4)

    conv = MyConv(8, 32)

    with pytest.raises(ValueError, match="valid indices"):
        edge_index = torch.tensor([[-1, 1, 2, 2], [0, 0, 1, 1]])
        conv(x, edge_index, value)

    with pytest.raises(ValueError, match="valid indices"):
        edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
        conv(x, edge_index, value)


def test_my_conv_jittable():
    x1 = torch.randn(4, 8)
    x2 = torch.randn(2, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    value = torch.randn(row.size(0))
    adj = SparseTensor(row=row, col=col, value=value, sparse_sizes=(4, 4))

    conv = MyConv(8, 32)
    out = conv(x1, edge_index, value)

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


@pytest.mark.parametrize('aggr', ['add', 'sum', 'mean', 'min', 'max', 'mul'])
def test_my_conv_aggr(aggr):
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_weight = torch.randn(edge_index.size(1))

    conv = MyConv(8, 32, aggr=aggr)
    out = conv(x, edge_index, edge_weight)
    assert out.size() == (4, 32)


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


class MyMultipleAggrConv(MessagePassing):
    def __init__(self, **kwargs):
        super().__init__(aggr=['add', 'mean', 'max'], **kwargs)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        # propagate_type: (x: Tensor)
        return self.propagate(edge_index, x=x, size=None)


@pytest.mark.parametrize('multi_aggr_tuple', [
    (dict(mode='cat'), 3),
    (dict(mode='proj', mode_kwargs=dict(in_channels=16, out_channels=16)), 1)
])
def test_my_multiple_aggr_conv(multi_aggr_tuple):
    # The 'cat' combine mode will expand the output dimensions by
    # the number of aggregators which is 3 here, while the 'proj'
    # mode keeps output dimensions unchanged.
    aggr_kwargs, expand = multi_aggr_tuple
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))

    conv = MyMultipleAggrConv(aggr_kwargs=aggr_kwargs)
    out = conv(x, edge_index)
    assert out.size() == (4, 16 * expand)
    assert torch.allclose(conv(x, adj.t()), out)


def test_my_multiple_aggr_conv_jittable():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))

    conv = MyMultipleAggrConv()
    out = conv(x, edge_index)

    t = '(Tensor, Tensor) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert torch.allclose(jit(x, edge_index), out)

    t = '(Tensor, SparseTensor) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert torch.allclose(jit(x, adj.t()), out)


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


class MyEdgeConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        # edge_updater_type: (x: Tensor)
        edge_attr = self.edge_updater(edge_index, x=x)

        # propagate_type: (edge_attr: Tensor)
        return self.propagate(edge_index, edge_attr=edge_attr,
                              size=(x.size(0), x.size(0)))

    def edge_update(self, x_j: Tensor, x_i: Tensor) -> Tensor:
        return x_j - x_i

    def message(self, edge_attr: Tensor) -> Tensor:
        return edge_attr


def test_my_edge_conv():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))

    expected = scatter(x[row] - x[col], col, dim=0, dim_size=4, reduce='add')

    conv = MyEdgeConv()
    out = conv(x, edge_index)
    assert out.size() == (4, 16)
    assert torch.allclose(out, expected)
    assert torch.allclose(conv(x, adj.t()), out)


def test_my_edge_conv_jittable():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))

    conv = MyEdgeConv()
    out = conv(x, edge_index)

    t = '(Tensor, Tensor) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert torch.allclose(jit(x, edge_index), out)

    t = '(Tensor, SparseTensor) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert torch.allclose(jit(x, adj.t()), out)


num_pre_hook_calls = 0
num_hook_calls = 0


def test_message_passing_hooks():
    conv = MyConv(8, 32)

    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    value = torch.randn(row.size(0))
    adj = SparseTensor(row=row, col=col, value=value, sparse_sizes=(4, 4))

    def pre_hook(module, inputs):
        assert module == conv
        global num_pre_hook_calls
        num_pre_hook_calls += 1
        return inputs

    def hook(module, inputs, output):
        assert module == conv
        global num_hook_calls
        num_hook_calls += 1
        return output

    handle1 = conv.register_propagate_forward_pre_hook(pre_hook)
    assert len(conv._propagate_forward_pre_hooks) == 1
    handle2 = conv.register_propagate_forward_hook(hook)
    assert len(conv._propagate_forward_hooks) == 1

    handle3 = conv.register_message_forward_pre_hook(pre_hook)
    assert len(conv._message_forward_pre_hooks) == 1
    handle4 = conv.register_message_forward_hook(hook)
    assert len(conv._message_forward_hooks) == 1

    handle5 = conv.register_aggregate_forward_pre_hook(pre_hook)
    assert len(conv._aggregate_forward_pre_hooks) == 1
    handle6 = conv.register_aggregate_forward_hook(hook)
    assert len(conv._aggregate_forward_hooks) == 1

    handle7 = conv.register_message_and_aggregate_forward_pre_hook(pre_hook)
    assert len(conv._message_and_aggregate_forward_pre_hooks) == 1
    handle8 = conv.register_message_and_aggregate_forward_hook(hook)
    assert len(conv._message_and_aggregate_forward_hooks) == 1

    out1 = conv(x, edge_index, value)
    assert num_pre_hook_calls == 3
    assert num_hook_calls == 3
    out2 = conv(x, adj.t())
    assert num_pre_hook_calls == 5
    assert num_hook_calls == 5
    assert torch.allclose(out1, out2)

    handle1.remove()
    assert len(conv._propagate_forward_pre_hooks) == 0
    handle2.remove()
    assert len(conv._propagate_forward_hooks) == 0

    handle3.remove()
    assert len(conv._message_forward_pre_hooks) == 0
    handle4.remove()
    assert len(conv._message_forward_hooks) == 0

    handle5.remove()
    assert len(conv._aggregate_forward_pre_hooks) == 0
    handle6.remove()
    assert len(conv._aggregate_forward_hooks) == 0

    handle7.remove()
    assert len(conv._message_and_aggregate_forward_pre_hooks) == 0
    handle8.remove()
    assert len(conv._message_and_aggregate_forward_hooks) == 0

    conv = MyEdgeConv()

    handle1 = conv.register_edge_update_forward_pre_hook(pre_hook)
    assert len(conv._edge_update_forward_pre_hooks) == 1
    handle2 = conv.register_edge_update_forward_hook(hook)
    assert len(conv._edge_update_forward_hooks) == 1

    out1 = conv(x, edge_index)
    assert num_pre_hook_calls == 6
    assert num_hook_calls == 6
    out2 = conv(x, adj.t())
    assert num_pre_hook_calls == 7
    assert num_hook_calls == 7
    assert torch.allclose(out1, out2)

    handle1.remove()
    assert len(conv._propagate_forward_pre_hooks) == 0
    handle2.remove()
    assert len(conv._propagate_forward_hooks) == 0


def test_modified_message_passing_hook():
    conv = MyConv(8, 32)

    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_weight = torch.randn(edge_index.size(1))

    out1 = conv(x, edge_index, edge_weight)

    def hook(module, inputs, output):
        assert len(inputs) == 1
        assert len(inputs[-1]) == 2
        assert 'x_j' in inputs[-1]
        assert 'edge_weight' in inputs[-1]
        return output + 1.

    conv.register_message_forward_hook(hook)

    out2 = conv(x, edge_index, edge_weight)
    assert not torch.allclose(out1, out2)


class MyDefaultArgConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')

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


def test_my_default_arg_conv_jittable():
    conv = MyDefaultArgConv()

    with pytest.raises(RuntimeError):  # This should not succeed in JIT mode.
        torch.jit.script(conv.jittable())


class MyMultipleOutputConv(MessagePassing):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        # propagate_type: (x: Tensor)
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_j: Tensor) -> Tuple[Tensor, Tensor]:
        return x_j, x_j

    def aggregate(self, inputs: Tuple[Tensor, Tensor],
                  index: Tensor) -> Tuple[Tensor, Tensor]:
        return (scatter(inputs[0], index, dim=0, reduce='add'),
                scatter(inputs[0], index, dim=0, reduce='mean'))

    def update(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        return inputs


def test_tuple_output():
    conv = MyMultipleOutputConv()

    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    out1 = conv(x, edge_index)
    assert isinstance(out1, tuple) and len(out1) == 2


def test_tuple_output_jittable():
    conv = MyMultipleOutputConv()

    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    out1 = conv(x, edge_index)
    assert isinstance(out1, tuple) and len(out1) == 2

    jit = torch.jit.script(conv.jittable())
    out2 = jit(x, edge_index)
    assert isinstance(out2, tuple) and len(out2) == 2
    assert torch.allclose(out1[0], out2[0])
    assert torch.allclose(out1[1], out2[1])


class MyExplainConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        return self.propagate(edge_index, x=x)


def test_explain_message():
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv = MyExplainConv()
    assert conv(x, edge_index).abs().sum() != 0.

    conv.explain = True

    with pytest.raises(ValueError, match="pre-defined 'edge_mask'"):
        conv(x, edge_index)

    conv._edge_mask = torch.tensor([0, 0, 0, 0], dtype=torch.float)
    conv._apply_sigmoid = False
    assert conv(x, edge_index).abs().sum() == 0.


class MyAggregatorConv(MessagePassing):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        # propagate_type: (x: TEnsor)
        return self.propagate(edge_index, x=x, size=None)


@pytest.mark.parametrize('aggr_module', [
    aggr.MeanAggregation(),
    aggr.SumAggregation(),
    aggr.MaxAggregation(),
    aggr.SoftmaxAggregation(),
    aggr.PowerMeanAggregation(),
    aggr.MultiAggregation(['mean', 'max'])
])
def test_message_passing_with_aggr_module(aggr_module):
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(4, 4))

    conv = MyAggregatorConv(aggr=aggr_module)
    assert isinstance(conv.aggr_module, aggr.Aggregation)
    out = conv(x, edge_index)
    assert out.size(0) == 4 and out.size(1) in {8, 16}
    assert torch.allclose(conv(x, adj.t()), out)


def test_message_passing_int32_edge_index():
    # Check that we can dispatch an int32 edge_index up to aggregation
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.int32)
    edge_weight = torch.randn(edge_index.shape[1])

    # Use a hook to promote the edge_index to long to workaround PyTorch CPU
    # backend restriction to int64 for the index.
    def cast_index_hook(module, inputs):
        input_dict = inputs[-1]
        input_dict['index'] = input_dict['index'].long()
        return (input_dict, )

    conv = MyConv(8, 32)
    conv.register_aggregate_forward_pre_hook(cast_index_hook)

    assert conv(x, edge_index, edge_weight).size() == (4, 32)
