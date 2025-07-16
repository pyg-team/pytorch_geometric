r"""Testing package.

This package provides helper methods and decorators to ease testing.
"""

from .asserts import assert_module
from .data import (FakeHeteroDataset, get_random_edge_index,
                   get_random_tensor_frame)
from .decorators import (disableExtensions, has_package, is_distributed_test,
                         is_full_test, minPython, noMac, noWindows, onlyCUDA,
                         onlyDistributedTest, onlyFullTest, onlyGraphviz,
                         onlyLinux, onlyNeighborSampler, onlyOnline, onlyRAG,
                         onlyXPU, withCUDA, withDevice, withHashTensor,
                         withMETIS, withoutExtensions, withPackage)
from .feature_store import MyFeatureStore
from .graph_store import MyGraphStore

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
