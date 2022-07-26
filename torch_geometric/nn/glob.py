from torch_geometric.deprecation import deprecated
from torch_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.nn.aggr import (
    AttentionalAggregation,
    GraphMultisetTransformer,
    Set2Set,
    SortAggr,
)

Set2Set = deprecated(
    details="use 'nn.aggr.Set2Set' instead",
    func_name='nn.glob.Set2Set',
)(Set2Set)

GraphMultisetTransformer = deprecated(
    details="use 'nn.aggr.GraphMultisetTransformer' instead",
    func_name='nn.glob.GraphMultisetTransformer',
)(GraphMultisetTransformer)


@deprecated(
    details="use 'nn.aggr.AttentionalAggregation' instead",
    func_name='nn.glob.GlobalAttention',
)
class GlobalAttention(AttentionalAggregation):
    def __call__(self, x, batch=None, size=None):
        return super().__call__(x, batch, dim_size=size)


@deprecated(
    details="use 'nn.aggr.SortAggr' instead",
    func_name='nn.glob.global_sort_pool',
)
def global_sort_pool(x, index, k):
    module = SortAggr(k=k)
    return module(x, index=index)


deprecated(
    details="use 'nn.pool.global_add_pool' instead",
    func_name='nn.glob.global_add_pool',
)(global_add_pool)

deprecated(
    details="use 'nn.pool.global_max_pool' instead",
    func_name='nn.glob.global_max_pool',
)(global_max_pool)

deprecated(
    details="use 'nn.pool.global_mean_pool' instead",
    func_name='nn.glob.global_mean_pool',
)(global_mean_pool)
