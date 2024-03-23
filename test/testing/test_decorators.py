from enum import Enum

import pytest
import torch

import torch_geometric.typing
from torch_geometric.testing import disableExtensions
from torch_geometric.testing.decorators import withDevice


def test_enable_extensions():
    try:
        import pyg_lib  # noqa
        assert torch_geometric.typing.WITH_PYG_LIB
    except (ImportError, OSError):
        assert not torch_geometric.typing.WITH_PYG_LIB

    try:
        import torch_scatter  # noqa
        assert torch_geometric.typing.WITH_TORCH_SCATTER
    except (ImportError, OSError):
        assert not torch_geometric.typing.WITH_TORCH_SCATTER

    try:
        import torch_sparse  # noqa
        assert torch_geometric.typing.WITH_TORCH_SPARSE
    except (ImportError, OSError):
        assert not torch_geometric.typing.WITH_TORCH_SPARSE


@disableExtensions
def test_disable_extensions():
    assert not torch_geometric.typing.WITH_PYG_LIB
    assert not torch_geometric.typing.WITH_TORCH_SCATTER
    assert not torch_geometric.typing.WITH_TORCH_SPARSE


backends = [
    'cpu', 'cuda', 'cudnn', 'mha', 'mps', 'mkl', 'mkldnn', 'nnpack', 'openmp',
    'opt_einsum', 'xeon', ''
]


class Processor(Enum):
    CPU = ['cpu']
    GPU = ['cuda:0']
    UNIFIED_MEMORY = ['mps']


@withDevice
def test_dummy(device):
    assert isinstance(device, torch.device)


processors = [value for processor in Processor for value in processor.value]


@pytest.mark.parametrize('backend', backends)
@pytest.mark.parametrize('processor', processors)
def test_withDevice(monkeypatch, backend, processor):
    monkeypatch.setenv('TORCH_BACKEND', backend)
    monkeypatch.setenv('TORCH_DEVICE', processor)
    if processor == Processor.GPU.value[0]:
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    elif processor == Processor.UNIFIED_MEMORY.value[0]:
        monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    else:
        monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)

    assert hasattr(test_dummy, 'pytestmark')
    assert len(test_dummy.pytestmark) == 1
    assert test_dummy.pytestmark[0].name == 'parametrize'
    assert test_dummy.pytestmark[0].args[0] == 'device'
    assert all([
        isinstance(arg, torch.device)
        for arg in test_dummy.pytestmark[0].args[1][0].values
    ])
