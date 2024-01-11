import inspect

import torch
from torch import Tensor

from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.nn.conv.utils.inspector import Inspector, Param


def test_inspector():
    inspector = Inspector(SAGEConv)
    assert str(inspector) == 'Inspector(SAGEConv)'
    assert inspector.implements('message')
    assert inspector.implements('message_and_aggregate')

    out = inspector.inspect(SAGEConv.message)
    assert len(out) == 2
    assert out[0] == {'x_j': Param(Tensor, inspect._empty)}
    assert out[1] == Tensor
    assert inspector.get_flat_arg_types(['message']) == {'x_j': Tensor}
    assert inspector.get_flat_arg_names(['message']) == {'x_j'}

    kwargs = {'x_j': torch.randn(5), 'x_i': torch.randn(5)}
    out = inspector.collect('message', kwargs)
    assert len(out) == 1
    assert torch.allclose(out['x_j'], kwargs['x_j'])

    inspector = Inspector(GATConv)
    assert str(inspector) == 'Inspector(GATConv)'
    assert inspector.implements('message')
    assert not inspector.implements('message_and_aggregate')

    out = inspector.inspect(GATConv.message)
    assert len(out) == 2
    assert out[0] == {
        'x_j': Param(Tensor, inspect._empty),
        'alpha': Param(Tensor, inspect._empty),
    }
    assert out[1] == Tensor
    assert inspector.get_flat_arg_types(['message']) == {
        'x_j': Tensor,
        'alpha': Tensor,
    }
    assert inspector.get_flat_arg_names(['message']) == {'x_j', 'alpha'}

    kwargs = {'x_j': torch.randn(5), 'alpha': torch.randn(5)}
    out = inspector.collect('message', kwargs)
    assert len(out) == 2
    assert torch.allclose(out['x_j'], kwargs['x_j'])
    assert torch.allclose(out['alpha'], kwargs['alpha'])
