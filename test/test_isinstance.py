import torch

from torch_geometric import is_torch_instance
from torch_geometric.testing import onlyLinux, withPackage


def test_basic():
    assert is_torch_instance(torch.nn.Linear(1, 1), torch.nn.Linear)


@onlyLinux
@withPackage('torch>=2.0.0')
def test_compile():
    model = torch.compile(torch.nn.Linear(1, 1))
    assert not isinstance(model, torch.nn.Linear)
    assert is_torch_instance(model, torch.nn.Linear)
