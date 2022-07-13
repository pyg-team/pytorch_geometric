from .glob import global_add_pool, global_mean_pool, global_max_pool
from .glob import GlobalPooling
from .sort import global_sort_pool
from .attention import GlobalAttention

__all__ = [
    'global_add_pool',
    'global_mean_pool',
    'global_max_pool',
    'GlobalPooling',
    'global_sort_pool',
    'GlobalAttention',
]

classes = __all__

from torch_geometric.deprecation import deprecated  # noqa
from torch_geometric.nn.aggr import Set2Set  # noqa
from torch_geometric.nn.aggr import GraphMultisetTransformer  # noqa

Set2Set = deprecated(
    details="use 'nn.aggr.Set2Set' instead",
    func_name='nn.glob.Set2Set',
)(Set2Set)

GraphMultisetTransformer = deprecated(
    details="use 'nn.aggr.GraphMultisetTransformer' instead",
    func_name='nn.glob.GraphMultisetTransformer')(GraphMultisetTransformer)
