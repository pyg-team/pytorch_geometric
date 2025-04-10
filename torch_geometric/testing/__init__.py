r"""Testing package.

This package provides helper methods and decorators to ease testing.
"""

from .decorators import (
    is_full_test,
    onlyFullTest,
    is_distributed_test,
    onlyDistributedTest,
    onlyLinux,
    noWindows,
    noMac,
    minPython,
    onlyCUDA,
    onlyXPU,
    onlyOnline,
    onlyGraphviz,
    onlyNeighborSampler,
    onlyRAG,
    has_package,
    withPackage,
    withDevice,
    withCUDA,
    withMETIS,
    withHashTensor,
    disableExtensions,
    withoutExtensions,
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
    'is_distributed_test',
    'onlyDistributedTest',
    'onlyLinux',
    'noWindows',
    'noMac',
    'minPython',
    'onlyCUDA',
    'onlyXPU',
    'onlyOnline',
    'onlyGraphviz',
    'onlyNeighborSampler',
    'onlyRAG',
    'has_package',
    'withPackage',
    'withDevice',
    'withCUDA',
    'withMETIS',
    'withHashTensor',
    'disableExtensions',
    'withoutExtensions',
    'assert_module',
    'MyFeatureStore',
    'MyGraphStore',
    'get_random_edge_index',
    'get_random_tensor_frame',
    'FakeHeteroDataset',
]
