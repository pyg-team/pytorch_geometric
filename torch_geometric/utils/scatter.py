from typing import Optional

import torch
from torch import Tensor

major, minor, _ = torch.__version__.split('.', maxsplit=2)
major, minor = int(major), int(minor)
has_pytorch112 = major > 1 or (major == 1 and minor >= 12)

if has_pytorch112:  # pragma: no cover

    class ScatterHelpers:
        @staticmethod
        def broadcast(src: Tensor, other: Tensor, dim: int) -> Tensor:
            dim = other.dim() + dim if dim < 0 else dim
            if src.dim() == 1:
                for _ in range(0, dim):
                    src = src.unsqueeze(0)
            for _ in range(src.dim(), other.dim()):
                src = src.unsqueeze(-1)
            src = src.expand(other.size())
            return src

        @staticmethod
        def generate_out(src: Tensor, index: Tensor, dim: int,
                         dim_size: Optional[int]) -> Tensor:
            size = list(src.size())
            if dim_size is not None:
                size[dim] = dim_size
            elif index.numel() > 0:
                size[dim] = int(index.max()) + 1
            else:
                size[dim] = 0
            return src.new_zeros(size)

        @staticmethod
        def scatter_mean(src: Tensor, index: Tensor, dim: int, out: Tensor,
                         dim_size: Optional[int]) -> Tensor:
            out.scatter_add_(dim, index, src)

            index_dim = dim
            if index_dim < 0:
                # `index_dim` counts axes from the begining (0)
                index_dim = index_dim + src.dim()
            if index.dim() <= index_dim:
                # in case `index` was broadcasted, `count` scatter should be
                # performed over the last axis
                index_dim = index.dim() - 1

            ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
            count = ScatterHelpers.generate_out(ones, index, index_dim,
                                                dim_size)
            count.scatter_add_(index_dim, index, ones)
            count[count < 1] = 1
            count = ScatterHelpers.broadcast(count, out, dim)
            if out.is_floating_point():
                out.true_divide_(count)
            else:
                out.div_(count, rounding_mode='floor')
            return out

    def scatter(src: Tensor, index: Tensor, dim: int = -1,
                out: Optional[Tensor] = None, dim_size: Optional[int] = None,
                reduce: str = 'sum') -> Tensor:
        reduce = 'sum' if reduce == 'add' else reduce
        reduce = 'prod' if reduce == 'mul' else reduce
        reduce = 'amin' if reduce == 'min' else reduce
        reduce = 'amax' if reduce == 'max' else reduce

        index = ScatterHelpers.broadcast(index, src, dim)
        include_self = out is not None

        if out is None:  # Generate `out` if not given:
            out = ScatterHelpers.generate_out(src, index, dim, dim_size)

        # explicit usage of `torch.scatter_add_` and switching to
        # `torch_scatter` implementation of mean algorithm comes with
        # significant performance boost.
        # TODO: use only `torch.scatter_reduce_` after performance issue will
        # be fixed on the PyTorch side.
        if reduce == 'mean':
            return ScatterHelpers.scatter_mean(src, index, dim, out, dim_size)
        elif reduce == 'sum':
            return out.scatter_add_(dim, index, src)
        return out.scatter_reduce_(dim, index, src, reduce,
                                   include_self=include_self)

else:
    import torch_scatter

    def scatter(src: Tensor, index: Tensor, dim: int = -1,
                out: Optional[Tensor] = None, dim_size: Optional[int] = None,
                reduce: str = 'sum') -> Tensor:
        return torch_scatter.scatter(src, index, dim, out, dim_size, reduce)
