from typing import Optional

import torch
from torch import Tensor

from torch_geometric.utils.scatter_reduce import scatter
from torch_geometric.utils.scatter_reduce.broadcast import broadcast


def scatter_softmax(src: Tensor, index: Tensor, dim: int = -1,
                    dim_size: Optional[int] = None) -> Tensor:
    if not torch.is_floating_point(src):
        raise ValueError('`scatter_softmax` can only be computed over tensors '
                         'with floating point data types.')

    index = broadcast(index, src, dim)

    max_value_per_index = scatter(src, index, dim=dim, dim_size=dim_size,
                                  reduce='max')
    max_per_src_element = max_value_per_index.gather(dim, index)

    recentered_scores = src - max_per_src_element
    recentered_scores_exp = recentered_scores.exp_()

    sum_per_index = scatter(recentered_scores_exp, index, dim,
                            dim_size=dim_size, reduce='sum')
    normalizing_constants = sum_per_index.gather(dim, index)

    return recentered_scores_exp.div(normalizing_constants)


def scatter_log_softmax(src: Tensor, index: Tensor, dim: int = -1,
                        dim_size: Optional[int] = None) -> Tensor:
    if not torch.is_floating_point(src):
        raise ValueError('`scatter_log_softmax` can only be computed over '
                         'tensors with floating point data types.')

    index = broadcast(index, src, dim)

    max_value_per_index = scatter(src, index, dim=dim, dim_size=dim_size,
                                  reduce='max')
    max_per_src_element = max_value_per_index.gather(dim, index)

    recentered_scores = src - max_per_src_element

    sum_per_index = scatter(recentered_scores.exp(), index, dim,
                            dim_size=dim_size, reduce='sum')
    eps = 1e-12  # for numerical stability
    normalizing_constants = sum_per_index.add_(eps).log_().gather(dim, index)

    return recentered_scores.sub_(normalizing_constants)
