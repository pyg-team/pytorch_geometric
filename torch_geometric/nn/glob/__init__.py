from .glob import global_add_pool, global_mean_pool, global_max_pool
from .sort import global_sort_pool
from .attention import GlobalAttention
from .gmt import GraphMultisetTransformer
from .equilibrium_aggregation import EquilibriumAggregation

__all__ = [
    'global_add_pool',
    'global_mean_pool',
    'global_max_pool',
    'global_sort_pool',
    'GlobalAttention',
    'GraphMultisetTransformer',
    'EquilibriumAggregation',
]

classes = __all__

from torch_geometric.deprecation import deprecated  # noqa
from torch_geometric.nn.aggr import Set2Set  # noqa

Set2Set = deprecated(
    details="use 'nn.aggr.Set2Set' instead",
    func_name='nn.glob.Set2Set',
)(Set2Set)
