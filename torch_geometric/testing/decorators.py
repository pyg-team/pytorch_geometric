import os
import sys
from importlib import import_module
from importlib.util import find_spec
from typing import Callable

import torch
from packaging.requirements import Requirement

from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE
from torch_geometric.visualization.graph import has_graphviz


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


def onlyLinux(func: Callable) -> Callable:
    r"""A decorator to specify that this function should only execute on
    Linux systems."""
    import pytest
    return pytest.mark.skipif(
        sys.platform != 'linux',
        reason="No Linux system",
    )(func)


def onlyPython(*args) -> Callable:
    r"""A decorator to skip tests for any Python version not listed."""
    def decorator(func: Callable) -> Callable:
        import pytest

        python_version = f'{sys.version_info.major}.{sys.version_info.minor}'
        return pytest.mark.skipif(
            python_version not in args,
            reason=f"Python {python_version} not supported",
        )(func)

    return decorator


def onlyCUDA(func: Callable) -> Callable:
    r"""A decorator to skip tests if CUDA is not found."""
    import pytest
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    )(func)


def onlyXPU(func: Callable) -> Callable:
    r"""A decorator to skip tests if XPU is not found."""
    import pytest
    try:
        import intel_extension_for_pytorch as ipex
        xpu_available = ipex.xpu.is_available()
    except ImportError:
        xpu_available = False
    return pytest.mark.skipif(
        not xpu_available,
        reason="XPU not available",
    )(func)


def onlyOnline(func: Callable):
    r"""A decorator to skip tests if there exists no connection to the
    internet."""
    import http.client as httplib

    import pytest

    has_connection = True
    connection = httplib.HTTPSConnection('8.8.8.8', timeout=5)
    try:
        connection.request('HEAD', '/')
    except Exception:
        has_connection = False
    finally:
        connection.close()

    return pytest.mark.skipif(
        not has_connection,
        reason="No internet connection",
    )(func)


def onlyGraphviz(func: Callable) -> Callable:
    r"""A decorator to specify that this function should only execute in case
    :obj:`graphviz` is installed."""
    import pytest
    return pytest.mark.skipif(
        not has_graphviz(),
        reason="Graphviz not installed",
    )(func)


def onlyNeighborSampler(func: Callable):
    r"""A decorator to skip tests if no neighborhood sampler package is
    installed."""
    import pytest
    return pytest.mark.skipif(
        not WITH_PYG_LIB and not WITH_TORCH_SPARSE,
        reason="No neighbor sampler installed",
    )(func)


def withPackage(*args) -> Callable:
    r"""A decorator to skip tests if certain packages are not installed.
    Also supports version specification."""
    def is_installed(package: str) -> bool:
        if '|' in package:
            return any(is_installed(p) for p in package.split('|'))

        req = Requirement(package)
        if find_spec(req.name) is None:
            return False
        module = import_module(req.name)
        if not hasattr(module, '__version__'):
            return True

        version = module.__version__
        # `req.specifier` does not support `.dev` suffixes, e.g., for
        # `pyg_lib==0.1.0.dev*`, so we manually drop them:
        if '.dev' in version:
            version = '.'.join(version.split('.dev')[:-1])

        return version in req.specifier

    na_packages = set(package for package in args if not is_installed(package))

    def decorator(func: Callable) -> Callable:
        import pytest
        return pytest.mark.skipif(
            len(na_packages) > 0,
            reason=f"Package(s) {na_packages} are not installed",
        )(func)

    return decorator


def withCUDA(func: Callable):
    r"""A decorator to test both on CPU and CUDA (if available)."""
    import pytest

    devices = [torch.device('cpu')]
    if torch.cuda.is_available():
        devices.append(torch.device('cuda:0'))

    return pytest.mark.parametrize('device', devices)(func)


def disableExtensions(func: Callable):
    r"""A decorator to temporarily disable the usage of the
    :obj:`torch_scatter`, :obj:`torch_sparse` and :obj:`pyg_lib` extension
    packages."""
    import pytest

    return pytest.mark.usefixtures('disable_extensions')(func)
