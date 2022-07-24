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


def global_add_pool(x: Tensor, batch: Optional[Tensor],
                    size: Optional[int] = None) -> Tensor:
    r"""Returns batch-wise graph-level-outputs by adding node features
    across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by
    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathbf{x}_n
    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
    """
    return sum_aggr(x, batch, dim_size=size)


mean_aggr = MeanAggregation()


def global_mean_pool(x: Tensor, batch: Optional[Tensor],
                     size: Optional[int] = None) -> Tensor:
    r"""Returns batch-wise graph-level-outputs by averaging node features
    across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by
    .. math::
        \mathbf{r}_i = \frac{1}{N_i} \sum_{n=1}^{N_i} \mathbf{x}_n
    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
    """
    return mean_aggr(x, batch, dim_size=size)


max_aggr = MaxAggregation()


def global_max_pool(x: Tensor, batch: Optional[Tensor],
                    size: Optional[int] = None) -> Tensor:
    r"""Returns batch-wise graph-level-outputs by taking the channel-wise
    maximum across the node dimension, so that for a single graph
    :math:`\mathcal{G}_i` its output is computed by
    .. math::
        \mathbf{r}_i = \mathrm{max}_{n=1}^{N_i} \, \mathbf{x}_n
    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example.
        size (int, optional): Batch-size :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
    """
    return max_aggr(x, batch, dim_size=size)


Set2Set = deprecated(
    details="use 'nn.aggr.Set2Set' instead",
    func_name='nn.glob.Set2Set',
)(Set2Set)

GraphMultisetTransformer = deprecated(
    details="use 'nn.aggr.GraphMultisetTransformer' instead",
    func_name='nn.glob.GraphMultisetTransformer',
)(GraphMultisetTransformer)


@deprecated(
    details="use 'nn.aggr.SortAggr' instead",
    func_name='nn.glob.global_sort_pool',
)
def global_sort_pool(x, index, k):
    module = SortAggr(k=k)
    return module(x, index=index)


@deprecated(
    details="use 'nn.aggr.AttentionalAggregation' instead",
    func_name='nn.glob.GlobalAttention',
)
class GlobalAttention(AttentionalAggregation):
    def __call__(self, x, batch=None, size=None):
        return super().__call__(x, batch, dim_size=size)
