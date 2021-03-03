from .meta import MetaLayer
from .data_parallel import DataParallel
from .reshape import Reshape
from .conv import *  # noqa
from .norm import *  # noqa
from .glob import *  # noqa
from .pool import *  # noqa
from .unpool import *  # noqa
from .dense import *  # noqa

# Lazy-load models to avoid loading unneeded dependencies
from torch_geometric.utils.lazy_loader import LazyLoader
models = LazyLoader("models", globals(), "torch_geometric.nn.models")

__all__ = [
    'MetaLayer',
    'DataParallel',
    'Reshape',
]
