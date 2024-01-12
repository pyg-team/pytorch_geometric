import inspect

import torch
from torch import Tensor

from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.nn.conv.utils.inspector import (
    Inspector,
    Parameter,
    Signature,
)


def test_inspector_sage_conv() -> None:
    inspector = Inspector(SAGEConv)
    assert str(inspector) == 'Inspector(SAGEConv)'
    assert inspector.implements('message')
    assert inspector.implements('message_and_aggregate')

    out = inspector.inspect_signature(SAGEConv.message)
    assert isinstance(out, Signature)
    assert out.param_dict == {'x_j': Parameter('x_j', Tensor, inspect._empty)}
    assert out.return_type == Tensor
    assert inspector.get_flat_params(['message', 'message']) == [
        Parameter('x_j', Tensor, inspect._empty),
    ]
    assert inspector.get_flat_param_names(['message']) == ['x_j']

    kwargs = {'x_j': torch.randn(5), 'x_i': torch.randn(5)}
    data = inspector.collect_param_data('message', kwargs)
    assert len(data) == 1
    assert torch.allclose(data['x_j'], kwargs['x_j'])

    assert inspector.get_params_from_method_call(SAGEConv.propagate) == {
        'x': Parameter('x', 'OptPairTensor', inspect._empty),
    }


def test_inspector_gat_conv() -> None:
    inspector = Inspector(GATConv)
    assert str(inspector) == 'Inspector(GATConv)'
    assert inspector.implements('message')
    assert not inspector.implements('message_and_aggregate')

    out = inspector.inspect_signature(GATConv.message)
    assert isinstance(out, Signature)
    assert out.param_dict == {
        'x_j': Parameter('x_j', Tensor, inspect._empty),
        'alpha': Parameter('alpha', Tensor, inspect._empty),
    }
    assert out.return_type == Tensor
    assert inspector.get_flat_params(['message', 'message']) == [
        Parameter('x_j', Tensor, inspect._empty),
        Parameter('alpha', Tensor, inspect._empty),
    ]
    assert inspector.get_flat_param_names(['message']) == ['x_j', 'alpha']

    kwargs = {'x_j': torch.randn(5), 'alpha': torch.randn(5)}
    data = inspector.collect_param_data('message', kwargs)
    assert len(data) == 2
    assert torch.allclose(data['x_j'], kwargs['x_j'])
    assert torch.allclose(data['alpha'], kwargs['alpha'])

    assert inspector.get_params_from_method_call(SAGEConv.propagate) == {
        'x': Parameter('x', 'OptPairTensor', inspect._empty),
        'alpha': Parameter('alpha', 'Tensor', inspect._empty),
    }


def test_get_params_from_method_call() -> None:
    class FromMethodCall1:
        propagate_type = {'x': Tensor}

    inspector = Inspector(FromMethodCall1)
    assert inspector.get_params_from_method_call('propagate') == {
        'x': Parameter('x', Tensor, inspect._empty),
    }

    class FromMethodCall2:
        # propagate_type: (x: Tensor)
        pass

    inspector = Inspector(FromMethodCall2)
    assert inspector.get_params_from_method_call('propagate') == {
        'x': Parameter('x', 'Tensor', inspect._empty),
    }

    class FromMethodCall3:
        def forward(self) -> None:
            self.propagate(  # type: ignore
                torch.randn(5, 5),
                x=None,
                size=None,
            )

    inspector = Inspector(FromMethodCall3)
    exclude = [0, 'size']
    assert inspector.get_params_from_method_call('propagate', exclude) == {
        'x': Parameter('x', Tensor, inspect._empty),
    }

    class FromMethodCall4:
        pass

    inspector = Inspector(FromMethodCall4)
    assert inspector.get_params_from_method_call('propagate') == {}
