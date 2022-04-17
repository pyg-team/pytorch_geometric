from collections import OrderedDict

import torch
import torch.fx
from torch.nn import Dropout, Linear, ReLU
from torch_sparse import SparseTensor

from torch_geometric.nn import (
    GCNConv,
    JumpingKnowledge,
    MessagePassing,
    Sequential,
    global_mean_pool,
)


def test_sequential():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    batch = torch.zeros(4, dtype=torch.long)

    model = Sequential('x, edge_index', [
        (GCNConv(16, 64), 'x, edge_index -> x'),
        ReLU(inplace=True),
        (GCNConv(64, 64), 'x, edge_index -> x'),
        ReLU(inplace=True),
        Linear(64, 7),
    ])
    model.reset_parameters()

    assert len(model) == 5
    assert str(model) == (
        'Sequential(\n'
        '  (0): GCNConv(16, 64)\n'
        '  (1): ReLU(inplace=True)\n'
        '  (2): GCNConv(64, 64)\n'
        '  (3): ReLU(inplace=True)\n'
        '  (4): Linear(in_features=64, out_features=7, bias=True)\n'
        ')')

    assert isinstance(model[0], GCNConv)
    assert isinstance(model[1], ReLU)
    assert isinstance(model[2], GCNConv)
    assert isinstance(model[3], ReLU)
    assert isinstance(model[4], Linear)

    out = model(x, edge_index)
    assert out.size() == (4, 7)

    model = Sequential('x, edge_index, batch', [
        (Dropout(p=0.5), 'x -> x'),
        (GCNConv(16, 64), 'x, edge_index -> x1'),
        ReLU(inplace=True),
        (GCNConv(64, 64), 'x1, edge_index -> x2'),
        ReLU(inplace=True),
        (lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
        (JumpingKnowledge('cat', 64, num_layers=2), 'xs -> x'),
        (global_mean_pool, 'x, batch -> x'),
        Linear(2 * 64, 7),
    ])
    model.reset_parameters()

    out = model(x, edge_index, batch)
    assert out.size() == (1, 7)


def test_sequential_jittable():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    adj_t = SparseTensor(row=edge_index[0], col=edge_index[1]).t()

    model = Sequential('x: Tensor, edge_index: Tensor', [
        (GCNConv(16, 64).jittable(), 'x, edge_index -> x'),
        ReLU(inplace=True),
        (GCNConv(64, 64).jittable(), 'x, edge_index -> x'),
        ReLU(inplace=True),
        Linear(64, 7),
    ])
    torch.jit.script(model)(x, edge_index)

    model = Sequential('x: Tensor, edge_index: SparseTensor', [
        (GCNConv(16, 64).jittable(), 'x, edge_index -> x'),
        ReLU(inplace=True),
        (GCNConv(64, 64).jittable(), 'x, edge_index -> x'),
        ReLU(inplace=True),
        Linear(64, 7),
    ])
    torch.jit.script(model)(x, adj_t)


def symbolic_trace(module):
    class Tracer(torch.fx.Tracer):
        def is_leaf_module(self, module, *args, **kwargs) -> bool:
            return (isinstance(module, MessagePassing)
                    or super().is_leaf_module(module, *args, **kwargs))

    return torch.fx.GraphModule(module, Tracer().trace(module))


def test_sequential_tracable():
    model = Sequential('x, edge_index', [
        (GCNConv(16, 64), 'x, edge_index -> x1'),
        ReLU(inplace=True),
        (GCNConv(64, 64), 'x1, edge_index -> x2'),
        ReLU(inplace=True),
        (lambda x1, x2: x1 + x2, 'x1, x2 -> x'),
        Linear(64, 7),
    ])
    symbolic_trace(model)


def test_sequential_with_multiple_return_values():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    model = Sequential('x, edge_index', [
        (GCNConv(16, 32), 'x, edge_index -> x1'),
        (GCNConv(32, 64), 'x1, edge_index -> x2'),
        (lambda x1, x2: (x1, x2), 'x1, x2 -> x1, x2'),
    ])

    x1, x2 = model(x, edge_index)
    assert x1.size() == (4, 32)
    assert x2.size() == (4, 64)


def test_sequential_with_ordered_dict():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    model = Sequential(
        'x, edge_index', modules=OrderedDict([
            ('conv1', (GCNConv(16, 32), 'x, edge_index -> x')),
            ('conv2', (GCNConv(32, 64), 'x, edge_index -> x')),
        ]))

    assert isinstance(model.conv1, GCNConv)
    assert isinstance(model.conv2, GCNConv)

    x = model(x, edge_index)
    assert x.size() == (4, 64)
