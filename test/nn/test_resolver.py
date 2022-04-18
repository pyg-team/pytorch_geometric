import torch

from torch_geometric.nn.resolver import activation_resolver


def test_activation_resolver():
    assert isinstance(activation_resolver(torch.nn.ELU()), torch.nn.ELU)
    assert isinstance(activation_resolver(torch.nn.ReLU()), torch.nn.ReLU)
    assert isinstance(activation_resolver(torch.nn.PReLU()), torch.nn.PReLU)

    assert isinstance(activation_resolver('elu'), torch.nn.ELU)
    assert isinstance(activation_resolver('relu'), torch.nn.ReLU)
    assert isinstance(activation_resolver('prelu'), torch.nn.PReLU)
