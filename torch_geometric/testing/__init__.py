r"""Testing package.

This package provides helper methods and decorators to ease testing.
"""

from .decorators import (
    is_full_test,
    onlyFullTest,
    onlyLinux,
    noWindows,
    onlyPython,
    onlyCUDA,
    onlyXPU,
    onlyOnline,
    onlyGraphviz,
    onlyNeighborSampler,
    has_package,
    withPackage,
    withCUDA,
    disableExtensions,
)
from .asserts import assert_module
from .feature_store import MyFeatureStore
from .graph_store import MyGraphStore
from .data import (
    get_random_edge_index,
    get_random_tensor_frame,
    FakeHeteroDataset,
)

__all__ = [
    'is_full_test',
    'onlyFullTest',
    'onlyLinux',
    'noWindows',
    'onlyPython',
    'onlyCUDA',
    'onlyXPU',
    'onlyOnline',
    'onlyGraphviz',
    'onlyNeighborSampler',
    'has_package',
    'withPackage',
    'withCUDA',
    'disableExtensions',
    'assert_module',
    'MyFeatureStore',
    'MyGraphStore',
    'get_random_edge_index',
    'get_random_tensor_frame',
    'FakeHeteroDataset',
]
