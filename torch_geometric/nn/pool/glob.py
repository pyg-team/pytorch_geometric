from typing import Optional

from torch import Tensor

from torch_geometric.deprecation import deprecated
from torch_geometric.nn.aggr import (
    AttentionalAggregation,
    GraphMultisetTransformer,
    MaxAggregation,
    MeanAggregation,
    Set2Set,
    SortAggr,
    SumAggregation,
)

sum_aggr = SumAggregation()


def global_add_pool(x: Tensor, index: Optional[Tensor],
                    size: Optional[int] = None) -> Tensor:
    return sum_aggr(x, index, dim_size=size)


mean_aggr = MeanAggregation()


def global_mean_pool(x: Tensor, index: Optional[Tensor],
                     size: Optional[int] = None) -> Tensor:
    return mean_aggr(x, index, dim_size=size)


max_aggr = MaxAggregation()


def global_max_pool(x: Tensor, index: Optional[Tensor],
                    size: Optional[int] = None) -> Tensor:
    return max_aggr(x, index, dim_size=size)


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
