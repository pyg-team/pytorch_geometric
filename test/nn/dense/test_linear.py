import copy
import pytest
from itertools import product

import torch
from torch.nn import Linear as PTLinear
from torch.nn.parameter import UninitializedParameter

from torch_geometric.nn import Linear, HeteroLinear

weight_inits = ['glorot', 'kaiming_uniform', None]
bias_inits = ['zeros', None]


@pytest.mark.parametrize('weight,bias', product(weight_inits, bias_inits))
def test_linear(weight, bias):
    x = torch.randn(3, 4, 16)
    lin = Linear(16, 32, weight_initializer=weight, bias_initializer=bias)
    assert str(lin) == 'Linear(16, 32, bias=True)'
    assert lin(x).size() == (3, 4, 32)


@pytest.mark.parametrize('weight,bias', product(weight_inits, bias_inits))
def test_lazy_linear(weight, bias):
    x = torch.randn(3, 4, 16)
    lin = Linear(-1, 32, weight_initializer=weight, bias_initializer=bias)
    assert str(lin) == 'Linear(-1, 32, bias=True)'
    assert lin(x).size() == (3, 4, 32)
    assert str(lin) == 'Linear(16, 32, bias=True)'


@pytest.mark.parametrize('dim1,dim2', product([-1, 16], [-1, 16]))
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


@pytest.mark.parametrize('lazy', [True, False])
def test_identical_linear_default_initialization(lazy):
    x = torch.randn(3, 4, 16)

    torch.manual_seed(12345)
    lin1 = Linear(-1 if lazy else 16, 32)
    lin1(x)

    torch.manual_seed(12345)
    lin2 = PTLinear(16, 32)

    assert lin1.weight.tolist() == lin2.weight.tolist()
    assert lin1.bias.tolist() == lin2.bias.tolist()
    assert lin1(x).tolist() == lin2(x).tolist()


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
    assert torch.allclose(copied_lin.bias, lin.bias, atol=1e-6)


def test_hetero_linear():
    x = torch.randn((3, 16))
    node_type = torch.tensor([0, 1, 2])

    lin = HeteroLinear(in_channels=16, out_channels=32, num_node_types=3)
    assert str(lin) == 'HeteroLinear(16, 32, bias=True)'

    out = lin(x, node_type)
    assert out.size() == (3, 32)

    jit = torch.jit.script(lin)
    assert torch.allclose(jit(x, node_type), out)
