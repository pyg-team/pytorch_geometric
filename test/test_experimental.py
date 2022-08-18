import pytest

from torch_geometric import (
    disable_all_experimental_options,
    is_experimental_option_enabled,
    set_experimental_options,
)


@pytest.mark.parametrize('option', ['pytorch_scatter'])
def test_experimental(option):
    assert is_experimental_option_enabled(option) is False
    set_experimental_options({option: True})
    assert is_experimental_option_enabled(option) is True
    set_experimental_options({option: False})
    assert is_experimental_option_enabled(option) is False

    set_experimental_options({option: True})
    disable_all_experimental_options()
    assert is_experimental_option_enabled(option) is False

    with set_experimental_options({option: True}):
        assert is_experimental_option_enabled(option) is True
    assert is_experimental_option_enabled(option) is False

    set_experimental_options({option: True})
    with set_experimental_options({option: False}):
        assert is_experimental_option_enabled(option) is False
    assert is_experimental_option_enabled(option) is True
