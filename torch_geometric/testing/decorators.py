import os
import sys
import warnings
from importlib import import_module
from importlib.util import find_spec
from typing import Callable

import torch
from packaging.requirements import Requirement
from packaging.version import Version

import torch_geometric
from torch_geometric.typing import WITH_METIS, WITH_PYG_LIB, WITH_TORCH_SPARSE
from torch_geometric.visualization.graph import has_graphviz


def is_full_test() -> bool:
    r"""Whether to run the full but time-consuming test suite."""
    return os.getenv('FULL_TEST', '0') == '1'


def onlyFullTest(func: Callable) -> Callable:
    r"""A decorator to specify that this function belongs to the full test
    suite.
    """
    import pytest
    return pytest.mark.skipif(
        not is_full_test(),
        reason="Fast test run",
    )(func)


def is_distributed_test() -> bool:
    r"""Whether to run the distributed test suite."""
    return ((is_full_test() or os.getenv('DIST_TEST', '0') == '1')
            and sys.platform == 'linux' and has_package('pyg_lib'))


def onlyDistributedTest(func: Callable) -> Callable:
    r"""A decorator to specify that this function belongs to the distributed
    test suite.
    """
    import pytest
    return pytest.mark.skipif(
        not is_distributed_test(),
        reason="Fast test run",
    )(func)


def onlyLinux(func: Callable) -> Callable:
    r"""A decorator to specify that this function should only execute on
    Linux systems.
    """
    import pytest
    return pytest.mark.skipif(
        sys.platform != 'linux',
        reason="No Linux system",
    )(func)


def noWindows(func: Callable) -> Callable:
    r"""A decorator to specify that this function should not execute on
    Windows systems.
    """
    import pytest
    return pytest.mark.skipif(
        os.name == 'nt',
        reason="Windows system",
    )(func)


def noMac(func: Callable) -> Callable:
    r"""A decorator to specify that this function should not execute on
    macOS systems.
    """
    import pytest
    return pytest.mark.skipif(
        sys.platform == 'darwin',
        reason="macOS system",
    )(func)


def minPython(version: str) -> Callable:
    r"""A decorator to run tests on specific :python:`Python` versions only."""
    def decorator(func: Callable) -> Callable:
        import pytest

        major, minor = version.split('.')

        skip = False
        if sys.version_info.major < int(major):
            skip = True
        if (sys.version_info.major == int(major)
                and sys.version_info.minor < int(minor)):
            skip = True

        return pytest.mark.skipif(
            skip,
            reason=f"Python {version} required",
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
    return pytest.mark.skipif(
        not torch_geometric.is_xpu_available(),
        reason="XPU not available",
    )(func)


def onlyOnline(func: Callable) -> Callable:
    r"""A decorator to skip tests if there exists no connection to the
    internet.
    """
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
    :obj:`graphviz` is installed.
    """
    import pytest
    return pytest.mark.skipif(
        not has_graphviz(),
        reason="Graphviz not installed",
    )(func)


def onlyNeighborSampler(func: Callable) -> Callable:
    r"""A decorator to skip tests if no neighborhood sampler package is
    installed.
    """
    import pytest
    return pytest.mark.skipif(
        not WITH_PYG_LIB and not WITH_TORCH_SPARSE,
        reason="No neighbor sampler installed",
    )(func)


def has_package(package: str) -> bool:
    r"""Returns :obj:`True` in case :obj:`package` is installed."""
    if '|' in package:
        return any(has_package(p) for p in package.split('|'))

    req = Requirement(package)
    if find_spec(req.name) is None:
        return False

    try:
        module = import_module(req.name)
        if not hasattr(module, '__version__'):
            return True

        version = Version(module.__version__).base_version
        return version in req.specifier
    except Exception:
        return False


def withPackage(*args: str) -> Callable:
    r"""A decorator to skip tests if certain packages are not installed.
    Also supports version specification.
    """
    na_packages = {package for package in args if not has_package(package)}

    if len(na_packages) == 1:
        reason = f"Package {list(na_packages)[0]} not found"
    else:
        reason = f"Packages {na_packages} not found"

    def decorator(func: Callable) -> Callable:
        import pytest
        return pytest.mark.skipif(len(na_packages) > 0, reason=reason)(func)

    return decorator


def withCUDA(func: Callable) -> Callable:
    r"""A decorator to test both on CPU and CUDA (if available)."""
    import pytest

    devices = [pytest.param(torch.device('cpu'), id='cpu')]
    if torch.cuda.is_available():
        devices.append(pytest.param(torch.device('cuda:0'), id='cuda:0'))

    return pytest.mark.parametrize('device', devices)(func)


def withDevice(func: Callable) -> Callable:
    r"""A decorator to test on all available tensor processing devices."""
    import pytest

    devices = [pytest.param(torch.device('cpu'), id='cpu')]

    if torch.cuda.is_available():
        devices.append(pytest.param(torch.device('cuda:0'), id='cuda:0'))

    if torch_geometric.is_mps_available():
        devices.append(pytest.param(torch.device('mps:0'), id='mps'))

    if torch_geometric.is_xpu_available():
        devices.append(pytest.param(torch.device('xpu:0'), id='xpu'))

    # Additional devices can be registered through environment variables:
    device = os.getenv('TORCH_DEVICE')
    if device:
        backend = os.getenv('TORCH_BACKEND')
        if backend is None:
            warnings.warn(f"Please specify the backend via 'TORCH_BACKEND' in"
                          f"order to test against '{device}'")
        else:
            import_module(backend)
            devices.append(pytest.param(torch.device(device), id=device))

    return pytest.mark.parametrize('device', devices)(func)


def withMETIS(func: Callable) -> Callable:
    r"""A decorator to only test in case a valid METIS method is available."""
    import pytest

    with_metis = WITH_METIS

    if with_metis:
        try:  # Test that METIS can succesfully execute:
            # TODO Using `pyg-lib` metis partitioning leads to some weird bugs
            # in the # CI. As such, we require `torch-sparse` for now.
            rowptr = torch.tensor([0, 2, 4, 6])
            col = torch.tensor([1, 2, 0, 2, 1, 0])
            torch.ops.torch_sparse.partition(rowptr, col, None, 2, True)
        except Exception:
            with_metis = False

    return pytest.mark.skipif(
        not with_metis,
        reason="METIS not enabled",
    )(func)


def disableExtensions(func: Callable) -> Callable:
    r"""A decorator to temporarily disable the usage of the
    :obj:`torch_scatter`, :obj:`torch_sparse` and :obj:`pyg_lib` extension
    packages.
    """
    import pytest

    return pytest.mark.usefixtures('disable_extensions')(func)


def withoutExtensions(func: Callable) -> Callable:
    r"""A decorator to test both with and without the usage of extension
    packages such as :obj:`torch_scatter`, :obj:`torch_sparse` and
    :obj:`pyg_lib`.
    """
    import pytest

    return pytest.mark.parametrize(
        'without_extensions',
        ['enable_extensions', 'disable_extensions'],
        indirect=True,
    )(func)
