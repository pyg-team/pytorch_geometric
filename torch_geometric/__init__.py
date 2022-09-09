from types import ModuleType
from importlib import import_module

import torch_geometric.data
import torch_geometric.loader
import torch_geometric.transforms
import torch_geometric.utils
import torch_geometric.profile

from .seed import seed_everything
from .home import get_home_dir, set_home_dir
from .debug import is_debug_enabled, debug, set_debug
from .experimental import (is_experimental_mode_enabled, experimental_mode,
                           set_experimental_mode)


# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/
# python/util/lazy_loader.py
class LazyLoader(ModuleType):
    def __init__(self, local_name, parent_module_globals, name):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        super().__init__(name)

    def _load(self):
        module = import_module(self.__name__)
        self._parent_module_globals[self._local_name] = module
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)


datasets = LazyLoader('datasets', globals(), 'torch_geometric.datasets')
nn = LazyLoader('nn', globals(), 'torch_geometric.nn')
graphgym = LazyLoader('graphgym', globals(), 'torch_geometric.graphgym')

__version__ = '2.1.0'

__all__ = [
    'seed_everything',
    'get_home_dir',
    'set_home_dir',
    'is_debug_enabled',
    'debug',
    'set_debug',
    'is_experimental_mode_enabled',
    'experimental_mode',
    'set_experimental_mode',
    'torch_geometric',
    '__version__',
]
