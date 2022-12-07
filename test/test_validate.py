from torch_geometric import is_validation_enabled, set_validation, validate


def test_validate():
    try:
        prev = is_validation_enabled()
        set_validation(False)
        assert not is_validation_enabled()

        with validate(True):
            assert is_validation_enabled()

        assert not is_validation_enabled()
    finally:
        set_validation(prev)
