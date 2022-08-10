import warnings
from typing import Optional

import torch
from torch import Tensor

major, minor, _ = torch.__version__.split('.', maxsplit=2)
major, minor = int(major), int(minor)
if major > 1 or (major == 1 and minor >= 12):
    warnings.filterwarnings('ignore', '.*scatter_reduce.*')

    def scatter(src: Tensor, index: Tensor, dim: int = -1,
                out: Optional[Tensor] = None, dim_size: Optional[int] = None,
                reduce: str = 'sum') -> Tensor:

        reduce = 'sum' if reduce == 'add' else reduce
        reduce = 'prod' if reduce == 'mul' else reduce
        reduce = 'amin' if reduce == 'min' else reduce
        reduce = 'amax' if reduce == 'max' else reduce

        # Broadcast `index`:
        dim = src.dim() + dim if dim < 0 else dim
        if index.dim() == 1:
            for _ in range(0, dim):
                index = index.unsqueeze(0)
        for _ in range(index.dim(), src.dim()):
            index = index.unsqueeze(-1)
        index = index.expand(src.size())

        include_self = out is not None

        if out is None:  # Generate `out` if not given:
            size = list(src.size())
            if dim_size is not None:
                size[dim] = dim_size
            elif index.numel() > 0:
                size[dim] = int(index.max()) + 1
            else:
                size[dim] = 0
            out = src.new_zeros(size)

        out = out.scatter_reduce_(dim, index, src, reduce,
                                  include_self=include_self)

        # if not include_self:
        #     out.mark_dirty = False
        if out.grad_fn is not None:
            # print(out.__dict__)
            # print(out.__class__.__dict__.keys())
            # print(out.grad_fn)
            # print(out.grad_fn.__class__)
            # print(out.grad_fn.__class__.__dict__.keys())
            # print(torch.autograd.Function.__dict__.keys())

            print(torch.Tensor.scatter_reduce)
            print(torch.Tensor.scatter_reduce)

        return out

else:
    import torch_scatter

    def scatter(src: Tensor, index: Tensor, dim: int = -1,
                out: Optional[Tensor] = None, dim_size: Optional[int] = None,
                reduce: str = "sum") -> Tensor:
        return torch_scatter.scatter(src, index, dim, out, dim_size, reduce)
