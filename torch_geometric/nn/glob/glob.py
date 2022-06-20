from typing import List, Optional, Union

import torch
from torch import Tensor
from torch_scatter import scatter


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
    if batch is None:
        return x.sum(dim=-2, keepdim=x.dim() == 2)
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=-2, dim_size=size, reduce='add')


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
    if batch is None:
        return x.mean(dim=-2, keepdim=x.dim() == 2)
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=-2, dim_size=size, reduce='mean')


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
    if batch is None:
        return x.max(dim=-2, keepdim=x.dim() == 2)[0]
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=-2, dim_size=size, reduce='max')


class GlobalPooling(torch.nn.Module):
    r"""A global pooling module that wraps the usage of
    :meth:`~torch_geometric.nn.glob.global_add_pool`,
    :meth:`~torch_geometric.nn.glob.global_mean_pool` and
    :meth:`~torch_geometric.nn.glob.global_max_pool` into a single module.

    Args:
        aggr (string or List[str]): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            If given as a list, will make use of multiple aggregations in which
            different outputs will get concatenated in the last dimension.
    """
    def __init__(self, aggr: Union[str, List[str]]):
        super().__init__()

        self.aggrs = [aggr] if isinstance(aggr, str) else aggr

        assert len(self.aggrs) > 0
        assert len(set(self.aggrs) | {'sum', 'add', 'mean', 'max'}) == 4

    def forward(self, x: Tensor, batch: Optional[Tensor],
                size: Optional[int] = None) -> Tensor:
        """"""
        xs: List[Tensor] = []

        for aggr in self.aggrs:
            if aggr == 'sum' or aggr == 'add':
                xs.append(global_add_pool(x, batch, size))
            elif aggr == 'mean':
                xs.append(global_mean_pool(x, batch, size))
            elif aggr == 'max':
                xs.append(global_max_pool(x, batch, size))

        return xs[0] if len(xs) == 1 else torch.cat(xs, dim=-1)

    def __repr__(self) -> str:
        aggr = self.aggrs[0] if len(self.aggrs) == 1 else self.aggrs
        return f'{self.__class__.__name__}(aggr={aggr})'
