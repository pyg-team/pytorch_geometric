import pytest
import torch.nn as nn

import torch_geometric.contrib.nn.layers.activation as _act

get_activation_function = _act.get_activation_function


def test_get_activation_function_valid():
    activations = {
        'ReLU': nn.ReLU,
        'LeakyReLU': nn.LeakyReLU,
        'PReLU': nn.PReLU,
        'tanh': nn.Tanh,
        'SELU': nn.SELU,
        'ELU': nn.ELU,
        'Linear': type(lambda x: x),
        'GELU': nn.GELU,
    }
    for name, expected in activations.items():
        act = get_activation_function(name)

        if name == 'Linear':
            assert callable(act)
        else:
            assert isinstance(
                act, expected
            ), f"Activation for {name} is not instance of {expected}"


def test_get_activation_function_invalid():
    with pytest.raises(ValueError):
        get_activation_function("invalid_activation")
