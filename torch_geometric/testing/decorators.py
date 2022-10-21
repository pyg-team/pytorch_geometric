import os
import sys
from importlib import import_module
from importlib.util import find_spec
from typing import Callable

import torch
from packaging.requirements import Requirement


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


def onlyUnix(func: Callable) -> Callable:
    r"""A decorator to specify that this function should only execute on
    UNIX-like systems."""
    import pytest
    return pytest.mark.skipif(
        sys.platform == 'win32',
        reason="Windows not supported",
    )(func)


def withPython(*args) -> Callable:
    r"""A decorator to skip tests for any Python version not listed."""
    def decorator(func: Callable) -> Callable:
        import pytest

        python_version = f'{sys.version_info.major}.{sys.version_info.minor}'
        return pytest.mark.skipif(
            python_version not in args,
            reason=f"Python {python_version} not supported",
        )(func)

    return decorator


def withPackage(*args) -> Callable:
    r"""A decorator to skip tests if certain packages are not installed.
    Also supports version specification."""
    def is_installed(package: str) -> bool:
        req = Requirement(package)
        if find_spec(req.name) is None:
            return False
        module = import_module(req.name)
        if not hasattr(module, '__version__'):
            return True
        return module.__version__ in req.specifier

    na_packages = set(package for package in args if not is_installed(package))

    def decorator(func: Callable) -> Callable:
        import pytest
        return pytest.mark.skipif(
            len(na_packages) > 0,
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
