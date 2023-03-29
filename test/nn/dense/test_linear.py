import copy

import pytest
import torch
from torch.nn import Linear as PTLinear
from torch.nn.parameter import UninitializedParameter

from torch_geometric.nn import HeteroDictLinear, HeteroLinear, Linear
from torch_geometric.testing import is_full_test, withPackage

weight_inits = ['glorot', 'kaiming_uniform', None]
bias_inits = ['zeros', None]


@pytest.mark.parametrize('weight', weight_inits)
@pytest.mark.parametrize('bias', bias_inits)
def test_linear(weight, bias):
    x = torch.randn(3, 4, 16)
    lin = Linear(16, 32, weight_initializer=weight, bias_initializer=bias)
    assert str(lin) == 'Linear(16, 32, bias=True)'
    assert lin(x).size() == (3, 4, 32)


@pytest.mark.parametrize('weight', weight_inits)
@pytest.mark.parametrize('bias', bias_inits)
def test_lazy_linear(weight, bias):
    x = torch.randn(3, 4, 16)
    lin = Linear(-1, 32, weight_initializer=weight, bias_initializer=bias)
    assert str(lin) == 'Linear(-1, 32, bias=True)'
    assert lin(x).size() == (3, 4, 32)
    assert str(lin) == 'Linear(16, 32, bias=True)'


@pytest.mark.parametrize('dim1', [-1, 16])
@pytest.mark.parametrize('dim2', [-1, 16])
def test_load_lazy_linear(dim1, dim2):
    lin1 = Linear(dim1, 32)
    lin2 = Linear(dim1, 32)
    lin2.load_state_dict(lin1.state_dict())

    if dim1 != -1:
        assert torch.allclose(lin1.weight, lin2.weight)
        assert torch.allclose(lin1.bias, lin2.bias)
        assert not hasattr(lin1, '_hook')
        assert not hasattr(lin2, '_hook')
    else:
        assert isinstance(lin1.weight, UninitializedParameter)
        assert isinstance(lin2.weight, UninitializedParameter)
        assert hasattr(lin1, '_hook')
        assert hasattr(lin2, '_hook')

    with pytest.raises(RuntimeError, match="in state_dict"):
        lin1.load_state_dict({}, strict=True)
    lin1.load_state_dict({}, strict=False)


@pytest.mark.parametrize('lazy', [True, False])
def test_identical_linear_default_initialization(lazy):
    x = torch.randn(3, 4, 16)

    torch.manual_seed(12345)
    lin1 = Linear(-1 if lazy else 16, 32)
    lin1(x)

    torch.manual_seed(12345)
    lin2 = PTLinear(16, 32)

    assert torch.equal(lin1.weight, lin2.weight)
    assert torch.equal(lin1.bias, lin2.bias)
    assert torch.allclose(lin1(x), lin2(x))


@withPackage('torch<=1.12')
def test_copy_unintialized_parameter():
    weight = UninitializedParameter()
    with pytest.raises(Exception):
        copy.deepcopy(weight)


@pytest.mark.parametrize('lazy', [True, False])
def test_copy_linear(lazy):
    lin = Linear(-1 if lazy else 16, 32)

    copied_lin = copy.copy(lin)
    assert id(copied_lin) != id(lin)
    assert id(copied_lin.weight) == id(lin.weight)
    if not isinstance(copied_lin.weight, UninitializedParameter):
        assert copied_lin.weight.data_ptr() == lin.weight.data_ptr()
    assert id(copied_lin.bias) == id(lin.bias)
    assert copied_lin.bias.data_ptr() == lin.bias.data_ptr()

    copied_lin = copy.deepcopy(lin)
    assert id(copied_lin) != id(lin)
    assert id(copied_lin.weight) != id(lin.weight)
    if not isinstance(copied_lin.weight, UninitializedParameter):
        assert copied_lin.weight.data_ptr() != lin.weight.data_ptr()
        assert torch.allclose(copied_lin.weight, lin.weight)
    assert id(copied_lin.bias) != id(lin.bias)
    assert copied_lin.bias.data_ptr() != lin.bias.data_ptr()
    if int(torch.isnan(lin.bias).sum()) == 0:
        assert torch.allclose(copied_lin.bias, lin.bias)


def test_hetero_linear():
    x = torch.randn(3, 16)
    type_vec = torch.tensor([0, 1, 2])

    lin = HeteroLinear(16, 32, num_types=3)
    assert str(lin) == 'HeteroLinear(16, 32, num_types=3, bias=True)'

    out = lin(x, type_vec)
    assert out.size() == (3, 32)

    if is_full_test():
        jit = torch.jit.script(lin)
        assert torch.allclose(jit(x, type_vec), out)


def test_lazy_hetero_linear():
    x = torch.randn(3, 16)
    type_vec = torch.tensor([0, 1, 2])

    lin = HeteroLinear(-1, 32, num_types=3)
    assert str(lin) == 'HeteroLinear(-1, 32, num_types=3, bias=True)'

    out = lin(x, type_vec)
    assert out.size() == (3, 32)


def test_hetero_dict_linear():
    x_dict = {'v': torch.randn(3, 16), 'w': torch.randn(2, 8)}

    lin = HeteroDictLinear({'v': 16, 'w': 8}, 32)
    assert str(lin) == "HeteroDictLinear({'v': 16, 'w': 8}, 32, bias=True)"

    out_dict = lin(x_dict)
    assert len(out_dict) == 2
    assert out_dict['v'].size() == (3, 32)
    assert out_dict['w'].size() == (2, 32)

    x_dict = {'v': torch.randn(3, 16), 'w': torch.randn(2, 16)}

    lin = HeteroDictLinear(16, 32, types=['v', 'w'])
    assert str(lin) == "HeteroDictLinear({'v': 16, 'w': 16}, 32, bias=True)"

    out_dict = lin(x_dict)
    assert len(out_dict) == 2
    assert out_dict['v'].size() == (3, 32)
    assert out_dict['w'].size() == (2, 32)


def test_lazy_hetero_dict_linear():
    x_dict = {'v': torch.randn(3, 16), 'w': torch.randn(2, 8)}

    lin = HeteroDictLinear(-1, 32, types=['v', 'w'])
    assert str(lin) == "HeteroDictLinear({'v': -1, 'w': -1}, 32, bias=True)"

    out_dict = lin(x_dict)
    assert len(out_dict) == 2
    assert out_dict['v'].size() == (3, 32)
    assert out_dict['w'].size() == (2, 32)


@withPackage('pyg_lib')
@pytest.mark.parametrize('type_vec', [
    torch.tensor([0, 0, 1, 1, 2, 2]),
    torch.tensor([0, 1, 2, 0, 1, 2]),
])
def test_hetero_linear_sort(type_vec):
    x = torch.randn(type_vec.numel(), 16)

    lin = HeteroLinear(16, 32, num_types=3)
    out = lin(x, type_vec)

    for i in range(type_vec.numel()):
        node_type = int(type_vec[i])
        expected = x[i] @ lin.weight[node_type] + lin.bias[node_type]
        assert torch.allclose(out[i], expected, atol=1e-6)
