import torch

from torch_geometric import (
    is_torch_instance,
    is_torch_module_dict,
    is_torch_module_list,
)
from torch_geometric.testing import onlyLinux, withPackage


def test_basic():
    assert is_torch_instance(torch.nn.Linear(1, 1), torch.nn.Linear)


@onlyLinux
@withPackage('torch>=2.0.0')
def test_compile():
    model = torch.compile(torch.nn.Linear(1, 1))
    assert not isinstance(model, torch.nn.Linear)
    assert is_torch_instance(model, torch.nn.Linear)


def test_torch_module_dict():
    assert is_torch_module_dict(torch.nn.ModuleDict({}))
    assert is_torch_module_dict(
        torch.nn.ModuleDict({"a": torch.nn.Linear(1, 1)}))
    assert not is_torch_module_dict(
        torch.nn.ModuleList([torch.nn.Linear(1, 1)]))


def test_torch_module_list():
    assert is_torch_module_list(torch.nn.ModuleList([]))
    assert is_torch_module_list(torch.nn.ModuleList([torch.nn.Linear(1, 1)]))
    assert not is_torch_module_list(
        torch.nn.ModuleDict({"a": torch.nn.Linear(1, 1)}))
