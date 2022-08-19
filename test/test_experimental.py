import pytest

from torch_geometric import experimental_mode, is_experimental_mode_enabled


@pytest.mark.parametrize('options', [None, 'scatter_reduce'])
def test_experimental_mode(options):
    assert is_experimental_mode_enabled(options) is False
    with experimental_mode(options):
        assert is_experimental_mode_enabled(options) is True
    assert is_experimental_mode_enabled(options) is False
