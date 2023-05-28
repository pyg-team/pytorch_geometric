import torch_geometric.typing
from torch_geometric.testing import disableExtensions


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
