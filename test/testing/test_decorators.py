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
    'opt_einsum', 'xeon', None
]


@withDevice
def test_dummy(device):
    assert isinstance(device, torch.device)


@pytest.mark.parametrize('backend', backends)
def test_withDevice(monkeypatch, backend):
    monkeypatch.setenv('TORCH_BACKEND', backend)
    if backend == 'cuda':
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    elif backend == 'mps':
        monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    else:
        monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)

    # Check if the test_dummy function has been correctly parameterized
    assert hasattr(test_dummy, 'pytestmark')
    assert len(test_dummy.pytestmark) == 1
    assert test_dummy.pytestmark[0].name == 'parametrize'
    assert test_dummy.pytestmark[0].args[0] == 'device'
    assert all([
        isinstance(arg, torch.device)
        for arg in test_dummy.pytestmark[0].args[1][0].values
    ])
