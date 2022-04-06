import os
from importlib.util import find_spec
from typing import Callable

import torch


def is_full_test() -> bool:
    r"""Whether to run the full but time-consuming test suite."""
    return os.getenv('FULL_TEST', '0') == '1'


def onlyFullTest(func: Callable) -> Callable:
    r"""A decorator to specify that this function belongs to the full test
    suite."""
    import pytest
    return pytest.mark.skipif(
        not is_full_test(),
        reason="Fast test run",
    )(func)


def withPackage(*args) -> Callable:
    r"""A decorator to skip tests if certain packages are not installed."""
    na_packages = set(arg for arg in args if find_spec(arg) is None)

    def decorator(func: Callable) -> Callable:
        import pytest

        return pytest.mark.skipif(
            not is_full_test() and len(na_packages) > 0,
            reason=f"Package(s) {na_packages} are not installed",
        )(func)

    return decorator


def withCUDA(func: Callable) -> Callable:
    r"""A decorator to skip tests if CUDA is not found."""
    import pytest
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    )(func)
