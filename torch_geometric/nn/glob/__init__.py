from .glob import (
    GlobalPooling,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

__all__ = [
    'global_add_pool',
    'global_mean_pool',
    'global_max_pool',
    'GlobalPooling',
]

classes = __all__

from torch_geometric.deprecation import deprecated  # noqa
from torch_geometric.nn.aggr import AttentionalAggregation  # noqa
from torch_geometric.nn.aggr import GraphMultisetTransformer  # noqa
from torch_geometric.nn.aggr import SortAggr  # noqa
from torch_geometric.nn.aggr import Set2Set  # noqa

Set2Set = deprecated(
    details="use 'nn.aggr.Set2Set' instead",
    func_name='nn.glob.Set2Set',
)(Set2Set)

GraphMultisetTransformer = deprecated(
    details="use 'nn.aggr.GraphMultisetTransformer' instead",
    func_name='nn.glob.GraphMultisetTransformer',
)(GraphMultisetTransformer)


@deprecated(
    details="use 'nn.aggr.GlobalSortAggr' instead",
    func_name='nn.glob.global_sort_pool',
)
def global_sort_pool(x, index, k):
    module = SortAggr(k=k)
    return module(x, index=index)


@deprecated(
    details="use 'nn.aggr.GlobalAttention' instead",
    func_name='nn.glob.GlobalAttention',
)
class GlobalAttention(AttentionalAggregation):
    def __call__(self, x, batch=None, size=None):
        return super().__call__(x, batch, dim_size=size)
