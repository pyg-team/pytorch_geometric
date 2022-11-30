from torch_geometric import (
    is_validation_enabled,
    optional_validation,
    set_validation,
)


def test_validate():
    try:
        prev = is_validation_enabled()
        set_validation(False)
        assert not is_validation_enabled()

        with optional_validation(True):
            assert is_validation_enabled()

        assert not is_validation_enabled()
    finally:
        set_validation(prev)
