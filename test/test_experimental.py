import pytest

from torch_geometric import (
    experimental_mode,
    is_experimental_mode_enabled,
    set_experimental_mode,
)


@pytest.mark.parametrize('options', [None, 'scatter_reduce'])
def test_experimental_mode(options):
    assert is_experimental_mode_enabled(options) is False
    with experimental_mode(options):
        assert is_experimental_mode_enabled(options) is True
    assert is_experimental_mode_enabled(options) is False

    with set_experimental_mode(True, options):
        assert is_experimental_mode_enabled(options) is True
    assert is_experimental_mode_enabled(options) is False

    with set_experimental_mode(False, options):
        assert is_experimental_mode_enabled(options) is False
    assert is_experimental_mode_enabled(options) is False

    set_experimental_mode(True, options)
    assert is_experimental_mode_enabled(options) is True
    set_experimental_mode(False, options)
    assert is_experimental_mode_enabled(options) is False
