from torch_geometric.deprecation import deprecated
from torch_geometric.nn.aggr import (
    AttentionalAggregation,
    GraphMultisetTransformer,
    Set2Set,
    SortAggr,
)

Set2Set = deprecated(
    details="use 'nn.aggr.Set2Set' instead",
    func_name='nn.pool.Set2Set',
)(Set2Set)

GraphMultisetTransformer = deprecated(
    details="use 'nn.aggr.GraphMultisetTransformer' instead",
    func_name='nn.pool.GraphMultisetTransformer',
)(GraphMultisetTransformer)


@deprecated(
    details="use 'nn.aggr.SortAggr' instead",
    func_name='nn.pool.global_sort_pool',
)
def global_sort_pool(x, index, k):
    module = SortAggr(k=k)
    return module(x, index=index)


@deprecated(
    details="use 'nn.aggr.AttentionalAggregation' instead",
    func_name='nn.pool.GlobalAttention',
)
class GlobalAttention(AttentionalAggregation):
    def __call__(self, x, batch=None, size=None):
        return super().__call__(x, batch, dim_size=size)
