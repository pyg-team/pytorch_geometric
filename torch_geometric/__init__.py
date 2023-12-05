from .edge_index import EdgeIndex

import torch_geometric.utils
import torch_geometric.data
import torch_geometric.sampler
import torch_geometric.loader
import torch_geometric.transforms
import torch_geometric.datasets
import torch_geometric.nn
import torch_geometric.explain
import torch_geometric.profile

from .seed import seed_everything
from .home import get_home_dir, set_home_dir
from .compile import compile
from .isinstance import is_torch_instance
from .debug import is_debug_enabled, debug, set_debug
from .experimental import (is_experimental_mode_enabled, experimental_mode,
                           set_experimental_mode)
from .lazy_loader import LazyLoader

contrib = LazyLoader('contrib', globals(), 'torch_geometric.contrib')
graphgym = LazyLoader('graphgym', globals(), 'torch_geometric.graphgym')

__version__ = '2.4.0'

__all__ = [
    'EdgeIndex',
    'seed_everything',
    'get_home_dir',
    'set_home_dir',
    'compile',
    'is_torch_instance',
    'is_debug_enabled',
    'debug',
    'set_debug',
    'is_experimental_mode_enabled',
    'experimental_mode',
    'set_experimental_mode',
    'torch_geometric',
    '__version__',
]
