from .meta import MetaLayer
from .reshape import Reshape
from .sequential import Sequential
from .data_parallel import DataParallel
from .to_hetero_transformer import to_hetero
from .to_hetero_with_bases_transformer import to_hetero_with_bases
from .encoding import PositionalEncoding

from .aggr import *  # noqa
from .conv import *  # noqa
from .pool import *  # noqa
from .glob import *  # noqa
from .norm import *  # noqa
from .unpool import *  # noqa
from .dense import *  # noqa
from .models import *  # noqa
from .functional import *  # noqa

__all__ = [
    'MetaLayer',
    'Reshape',
    'Sequential',
    'DataParallel',
    'to_hetero',
    'to_hetero_with_bases',
    'PositionalEncoding',
]

from torch_geometric.deprecation import deprecated
from .models.explainer import Explainer, to_captum
from .models.gnn_explainer import GNNExplainer

Explainer = deprecated("Will move to torch_geometric.explain")(
    Explainer)  # noqa
to_captum = deprecated("Will move to torch_geometric.explain")(
    to_captum)  # noqa
GNNExplainer = deprecated("Will move to torch_geometric.explain")(
    GNNExplainer)  # noqa
