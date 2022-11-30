from torch_geometric import (
    is_validation_enabled,
    optional_validation,
    set_validation,
)


def test_validate():
    set_validation(False)
    assert not is_validation_enabled()

    with optional_validation(True):
        assert is_validation_enabled()

    assert not is_validation_enabled()
