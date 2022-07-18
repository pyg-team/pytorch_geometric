from torch_geometric.deprecation import deprecated
from torch_geometric.nn.aggr import (SumAggregation, MeanAggregation,
                                     MaxAggregation, AttentionalAggregation,
                                     GraphMultisetTransformer, Set2Set,
                                     SortAggr)
from torch_geometric.nn.resolver import aggregation_resolver

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


sum_aggr = SumAggregation()


@deprecated(
    details="use 'nn.aggr.SumAggregation' instead",
    func_name='nn.glob.global_add_pool',
)
def global_add_pool(x, index, size):
    return sum_aggr(x, index, dim_size=size)


mean_aggr = MeanAggregation()


@deprecated(
    details="use 'nn.aggr.MeanAggregation' instead",
    func_name='nn.glob.global_mean_pool',
)
def global_mean_pool(x, index, size):
    return mean_aggr(x, index, dim_size=size)


max_aggr = MaxAggregation()


@deprecated(
    details="use 'nn.aggr.MaxAggregation' instead",
    func_name='nn.glob.global_max_pool',
)
def global_max_pool(x, index, size):
    return max_aggr(x, index, dim_size=size)


@deprecated(
    details="use 'nn.aggr' operators instead instead",
    func_name='nn.glob.GlobalPooling',
)
class GlobalPooling:
    def __init__(self, aggr):
        self.aggr = aggregation_resolver(aggr)

    def __call__(self, x, index, size):
        return self.aggr(x, index, dim_size=size)
