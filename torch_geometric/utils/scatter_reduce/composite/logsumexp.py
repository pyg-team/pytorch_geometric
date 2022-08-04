from typing import Optional

import torch
from torch import Tensor

from torch_geometric.utils.scatter_reduce import scatter
from torch_geometric.utils.scatter_reduce.broadcast import broadcast


def scatter_logsumexp(src: Tensor, index: Tensor, dim: int = -1,
                      out: Optional[Tensor] = None,
                      dim_size: Optional[int] = None) -> Tensor:
    if not torch.is_floating_point(src):
        raise ValueError('`scatter_logsumexp` can only be computed over '
                         'tensors with floating point data types.')

    index = broadcast(index, src, dim)

    if out is not None:
        dim_size = out.size(dim)
    else:
        if dim_size is None:
            dim_size = int(index.max()) + 1

    size = list(src.size())
    size[dim] = dim_size
    max_value_per_index = torch.full(size, float('-inf'), dtype=src.dtype,
                                     device=src.device)
    scatter(src, index, dim, max_value_per_index, dim_size=dim_size,
            reduce='max')
    max_per_src_element = max_value_per_index.gather(dim, index)
    recentered_score = src - max_per_src_element
    recentered_score.masked_fill_(torch.isnan(recentered_score), float('-inf'))

    if out is not None:
        out = out.sub_(max_value_per_index).exp_()

    sum_per_index = scatter(recentered_score.exp_(), index, dim, out, dim_size,
                            reduce='sum')
    eps = 1e-12  # for numerical stability

    return sum_per_index.add_(eps).log_().add_(max_value_per_index)
