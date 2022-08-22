import warnings
from typing import Optional

import torch
import torch_scatter
from torch import Tensor

import torch_geometric

major, minor, _ = torch.__version__.split('.', maxsplit=2)
major, minor = int(major), int(minor)
has_pytorch112 = major > 1 or (major == 1 and minor >= 12)


class ScatterImpl(torch.nn.Module):
    def forward(self, src: Tensor, index: Tensor, dim: int = -1,
                out: Optional[Tensor] = None, dim_size: Optional[int] = None,
                reduce: str = 'sum') -> Tensor:
        if torch_geometric.is_experimental_mode_enabled('scatter_reduce'):
            if not has_pytorch112:
                warnings.warn("Cannot use 'scatter_reduce' experimental mode "
                              "on PyTorch < 1.12")
                return torch_scatter.scatter(src, index, dim, out, dim_size,
                                             reduce)

            warnings.filterwarnings('ignore', '.*scatter_reduce.*')

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

            # TODO: For now, we need the clone here since otherwise, autograd
            # will complain about inplace modifications.
            # See: https://github.com/pyg-team/pytorch_geometric/pull/5120
            out = out.clone()

            return out

        return torch_scatter.scatter(src, index, dim, out, dim_size, reduce)


_scatter_impl = ScatterImpl()


def scatter(src: Tensor, index: Tensor, dim: int = -1,
            out: Optional[Tensor] = None, dim_size: Optional[int] = None,
            reduce: str = 'sum') -> Tensor:
    return _scatter_impl(src, index, dim, out, dim_size, reduce)
