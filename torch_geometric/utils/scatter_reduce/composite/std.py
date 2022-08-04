from typing import Optional

import torch
from torch import Tensor

from torch_geometric.utils.scatter_reduce import scatter
from torch_geometric.utils.scatter_reduce.broadcast import broadcast


def scatter_std(src: Tensor, index: Tensor, dim: int = -1,
                out: Optional[Tensor] = None, dim_size: Optional[int] = None,
                unbiased: bool = True) -> Tensor:
    if out is not None:
        dim_size = out.size(dim)

    if dim < 0:
        dim = src.dim() + dim

    count_dim = dim
    if index.dim() <= dim:
        count_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter(ones, index, count_dim, dim_size=dim_size, reduce='sum')

    index = broadcast(index, src, dim)
    tmp = scatter(src, index, dim, dim_size=dim_size, reduce='sum')
    count = broadcast(count, tmp, dim).clamp(1)
    mean = tmp.div(count)

    var = (src - mean.gather(dim, index))
    var = var * var
    out = scatter(var, index, dim, out, dim_size, reduce='sum')

    if unbiased:
        count = count.sub(1).clamp_(1)
    eps = 1e-6  # for numerical stability
    out = out.div(count + eps).sqrt()

    return out
