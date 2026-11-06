import inspect
import warnings
from types import SimpleNamespace
from typing import Any, Dict, Final, List, Optional, Set, Tuple, Union

import torch
from torch import Tensor

import torch_geometric.inspector
from torch_geometric.inspector import (
    Inspector,
    Parameter,
    Signature,
    eval_type,
)
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.typing import OptPairTensor


def test_eval_type() -> None:
    inspector = Inspector(SAGEConv)

    assert inspector.eval_type('Tensor') == Tensor
    assert inspector.eval_type('List[Tensor]') == List[Tensor]
    assert inspector.eval_type('Tuple[Tensor, int]') == Tuple[Tensor, int]
    assert inspector.eval_type('Tuple[int, ...]') == Tuple[int, ...]


def test_eval_type_without_deprecation_warning() -> None:
    # Omitting `type_params` in `typing._eval_type` is deprecated in Python
    # 3.13 and 3.14, and raises a `TypeError` from Python 3.15 onwards:
    _globals = {'Optional': Optional, 'Tensor': Tensor}
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        assert eval_type('Optional[Tensor]', _globals) == Optional[Tensor]


def test_eval_type_passes_type_params(monkeypatch) -> None:
    # Emulates Python 3.15, where `type_params` is a required argument:
    def _eval_type(value, globalns, localns, type_params):
        assert type_params == ()
        return Tensor

    module = torch_geometric.inspector
    monkeypatch.setattr(module, 'sys', SimpleNamespace(version_info=(3, 15)))
    monkeypatch.setattr(module.typing, '_eval_type', _eval_type)
    assert eval_type('Tensor', {'Tensor': Tensor}) == Tensor


def test_type_repr() -> None:
    inspector = Inspector(SAGEConv)

    assert inspector.type_repr(Any) == 'typing.Any'
    assert inspector.type_repr(Final) == 'typing.Final'
    assert inspector.type_repr(OptPairTensor) == (
        'Tuple[Tensor, Optional[Tensor]]')
    assert inspector.type_repr(
        Final[Optional[Tensor]]) == ('typing.Final[Optional[Tensor]]')
    assert inspector.type_repr(Union[None, Tensor]) == 'Optional[Tensor]'
    assert inspector.type_repr(Optional[Tensor]) == 'Optional[Tensor]'
    assert inspector.type_repr(Set[Tensor]) == 'typing.Set[Tensor]'
    assert inspector.type_repr(List) == 'List'
    assert inspector.type_repr(Tuple) == 'Tuple'
    assert inspector.type_repr(Set) == 'typing.Set'
    assert inspector.type_repr(Dict) == 'typing.Dict'
    assert inspector.type_repr(Dict[str, Tuple[Tensor, Tensor]]) == (  #
        'typing.Dict[str, Tuple[Tensor, Tensor]]')
    assert inspector.type_repr(Tuple[int, ...]) == 'Tuple[int, ...]'
    assert inspector.type_repr(Union[int, str, None]) == (  #
        'Union[int, str, None]')


def test_inspector_sage_conv() -> None:
    inspector = Inspector(SAGEConv)
    assert str(inspector) == 'Inspector(SAGEConv)'
    assert inspector.implements('message')
    assert inspector.implements('message_and_aggregate')

    out = inspector.inspect_signature(SAGEConv.message)
    assert isinstance(out, Signature)
    assert out.param_dict == {
        'x_j': Parameter('x_j', Tensor, 'Tensor', inspect._empty)
    }
    assert out.return_type == Tensor
    assert inspector.get_flat_params(['message', 'message']) == [
        Parameter('x_j', Tensor, 'Tensor', inspect._empty),
    ]
    assert inspector.get_flat_param_names(['message']) == ['x_j']

    kwargs = {'x_j': torch.randn(5), 'x_i': torch.randn(5)}
    data = inspector.collect_param_data('message', kwargs)
    assert len(data) == 1
    assert torch.allclose(data['x_j'], kwargs['x_j'])

    assert inspector.get_params_from_method_call(SAGEConv.propagate) == {
        'x': Parameter('x', OptPairTensor, 'OptPairTensor', inspect._empty),
    }


def test_inspector_gat_conv() -> None:
    inspector = Inspector(GATConv)
    assert str(inspector) == 'Inspector(GATConv)'
    assert inspector.implements('message')
    assert not inspector.implements('message_and_aggregate')

    out = inspector.inspect_signature(GATConv.message)
    assert isinstance(out, Signature)
    assert out.param_dict == {
        'x_j': Parameter('x_j', Tensor, 'Tensor', inspect._empty),
        'alpha': Parameter('alpha', Tensor, 'Tensor', inspect._empty),
    }
    assert out.return_type == Tensor
    assert inspector.get_flat_params(['message', 'message']) == [
        Parameter('x_j', Tensor, 'Tensor', inspect._empty),
        Parameter('alpha', Tensor, 'Tensor', inspect._empty),
    ]
    assert inspector.get_flat_param_names(['message']) == ['x_j', 'alpha']

    kwargs = {'x_j': torch.randn(5), 'alpha': torch.randn(5)}
    data = inspector.collect_param_data('message', kwargs)
    assert len(data) == 2
    assert torch.allclose(data['x_j'], kwargs['x_j'])
    assert torch.allclose(data['alpha'], kwargs['alpha'])

    assert inspector.get_params_from_method_call(SAGEConv.propagate) == {
        'x': Parameter('x', OptPairTensor, 'OptPairTensor', inspect._empty),
        'alpha': Parameter('alpha', Tensor, 'Tensor', inspect._empty),
    }


def test_get_params_from_method_call() -> None:
    class FromMethodCall1:
        propagate_type = {'x': Tensor}

    inspector = Inspector(FromMethodCall1)
    assert inspector.get_params_from_method_call('propagate') == {
        'x': Parameter('x', Tensor, 'Tensor', inspect._empty),
    }

    class FromMethodCall2:
        # propagate_type: (x: Tensor)
        pass

    inspector = Inspector(FromMethodCall2)
    assert inspector.get_params_from_method_call('propagate') == {
        'x': Parameter('x', Tensor, 'Tensor', inspect._empty),
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
        'x': Parameter('x', Tensor, 'Tensor', inspect._empty),
    }

    class FromMethodCall4:
        pass

    inspector = Inspector(FromMethodCall4)
    assert inspector.get_params_from_method_call('propagate') == {}
