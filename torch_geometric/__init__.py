import torch_geometric.data
import torch_geometric.datasets
import torch_geometric.explain
import torch_geometric.loader
import torch_geometric.nn
import torch_geometric.profile
import torch_geometric.sampler
import torch_geometric.transforms
import torch_geometric.utils

from .compile import compile
from .debug import debug, is_debug_enabled, set_debug
from .experimental import (
    experimental_mode,
    is_experimental_mode_enabled,
    set_experimental_mode,
)
from .home import get_home_dir, set_home_dir
from .lazy_loader import LazyLoader
from .seed import seed_everything

contrib = LazyLoader('contrib', globals(), 'torch_geometric.contrib')
graphgym = LazyLoader('graphgym', globals(), 'torch_geometric.graphgym')

__version__ = '2.4.0'

__all__ = [
    'seed_everything',
    'get_home_dir',
    'set_home_dir',
    'compile',
    'is_debug_enabled',
    'debug',
    'set_debug',
    'is_experimental_mode_enabled',
    'experimental_mode',
    'set_experimental_mode',
    'torch_geometric',
    '__version__',
]
