import warnings
from typing import Optional

import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric.typing import torch_scatter

major, minor, _ = torch.__version__.split('.', maxsplit=2)
major, minor = int(major), int(minor)
has_pytorch112 = major > 1 or (major == 1 and minor >= 12)

if has_pytorch112:  # pragma: no cover

    warnings.filterwarnings('ignore', '.*is in beta and the API may change.*')

    def scatter(src: Tensor, index: Tensor, dim: int = 0,
                dim_size: Optional[int] = None, reduce: str = 'sum') -> Tensor:
        r"""Reduces all values from the :obj:`src` tensor at the indices
        specified in the :obj:`index` tensor along a given dimension
        :obj:`dim`. See the `documentation
        <https://pytorch-scatter.readthedocs.io/en/latest/functions/
        scatter.html>`__ of the :obj:`torch_scatter` package for more
        information.

        Args:
            src (torch.Tensor): The source tensor.
            index (torch.Tensor): The index tensor.
            dim (int, optional): The dimension along which to index.
                (default: :obj:`0`)
            dim_size (int, optional): The size of the output tensor at
                dimension :obj:`dim`. If set to :obj:`None`, will create a
                minimal-sized output tensor according to
                :obj:`index.max() + 1`. (default: :obj:`None`)
            reduce (str, optional): The reduce operation (:obj:`"sum"`,
                :obj:`"mean"`, :obj:`"mul"`, :obj:`"min"` or :obj:`"max"`,
                :obj:`"any"`). (default: :obj:`"sum"`)
        """
        if index.dim() != 1:
            raise ValueError(f"The `index` argument must be one-dimensional "
                             f"(got {index.dim()} dimensions)")

        dim = src.dim() + dim if dim < 0 else dim

        if dim < 0 or dim >= src.dim():
            raise ValueError(f"The `dim` argument must lay between 0 and "
                             f"{src.dim() - 1} (got {dim})")

        if dim_size is None:
            dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

        # For now, we maintain various different code paths, based on whether
        # the input requires gradients and whether it lays on the CPU/GPU.
        # For example, `torch_scatter` is usually faster than
        # `torch.scatter_reduce` on GPU, while `torch.scatter_reduce` is faster
        # on CPU.
        # `torch.scatter_reduce` has a faster forward implementation for
        # "min"/"max" reductions since it does not compute additional arg
        # indices, but is therefore way slower in its backward implementation.
        # More insights can be found in `test/utils/test_scatter.py`.

        size = list(src.size())
        size[dim] = dim_size

        # For "any" reduction, we use regular `scatter_`:
        if reduce == 'any':
            index = broadcast(index, src, dim)
            return src.new_zeros(size).scatter_(dim, index, src)

        # For "sum" and "mean" reduction, we make use of `scatter_add_`:
        if reduce == 'sum' or reduce == 'add':
            index = broadcast(index, src, dim)
            return src.new_zeros(size).scatter_add_(dim, index, src)

        if reduce == 'mean':
            count = src.new_zeros(dim_size)
            count.scatter_add_(0, index, src.new_ones(src.size(dim)))
            count = count.clamp(min=1)

            index = broadcast(index, src, dim)
            out = src.new_zeros(size).scatter_add_(dim, index, src)

            return out / broadcast(count, out, dim)

        # For "min" and "max" reduction, we prefer `scatter_reduce_` on CPU or
        # in case the input does not require gradients:
        if reduce == 'min' or reduce == 'max':
            if (not torch_geometric.typing.WITH_TORCH_SCATTER
                    or not src.is_cuda or not src.requires_grad):

                if src.is_cuda and src.requires_grad:
                    warnings.warn(f"The usage of `scatter(reduce='{reduce}')` "
                                  f"can be accelerated via the 'torch-scatter'"
                                  f" package, but it was not found")

                index = broadcast(index, src, dim)
                return src.new_zeros(size).scatter_reduce_(
                    dim, index, src, reduce=f'a{reduce}', include_self=False)

            return torch_scatter.scatter(src, index, dim, dim_size=dim_size,
                                         reduce=reduce)

        # For "mul" reduction, we prefer `scatter_reduce_` on CPU:
        if reduce == 'mul':
            if (not torch_geometric.typing.WITH_TORCH_SCATTER
                    or not src.is_cuda):

                if src.is_cuda:
                    warnings.warn(f"The usage of `scatter(reduce='{reduce}')` "
                                  f"can be accelerated via the 'torch-scatter'"
                                  f" package, but it was not found")

                index = broadcast(index, src, dim)
                # We initialize with `one` here to match `scatter_mul` output:
                return src.new_ones(size).scatter_reduce_(
                    dim, index, src, reduce='prod', include_self=True)

            return torch_scatter.scatter(src, index, dim, dim_size=dim_size,
                                         reduce='mul')

        raise ValueError(f"Encountered invalid `reduce` argument '{reduce}'")

else:  # pragma: no cover

    def scatter(src: Tensor, index: Tensor, dim: int = 0,
                dim_size: Optional[int] = None, reduce: str = 'sum') -> Tensor:
        r"""Reduces all values from the :obj:`src` tensor at the indices
        specified in the :obj:`index` tensor along a given dimension
        :obj:`dim`. See the `documentation
        <https://pytorch-scatter.readthedocs.io/en/latest/functions/
        scatter.html>`_ of the :obj:`torch_scatter` package for more
        information.

        Args:
            src (torch.Tensor): The source tensor.
            index (torch.Tensor): The index tensor.
            dim (int, optional): The dimension along which to index.
                (default: :obj:`0`)
            dim_size (int, optional): The size of the output tensor at
                dimension :obj:`dim`. If set to :obj:`None`, will create a
                minimal-sized output tensor according to
                :obj:`index.max() + 1`. (default: :obj:`None`)
            reduce (str, optional): The reduce operation (:obj:`"sum"`,
                :obj:`"mean"`, :obj:`"mul"`, :obj:`"min"` or :obj:`"max"`).
                (default: :obj:`"sum"`)
        """
        if reduce == 'any':
            dim = src.dim() + dim if dim < 0 else dim

            if dim_size is None:
                dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

            size = list(src.size())
            size[dim] = dim_size

            index = broadcast(index, src, dim)
            return src.new_zeros(size).scatter_(dim, index, src)

        if not torch_geometric.typing.WITH_TORCH_SCATTER:
            raise ImportError("'scatter' requires the 'torch-scatter' package")
        return torch_scatter.scatter(src, index, dim, dim_size=dim_size,
                                     reduce=reduce)


def broadcast(src: Tensor, ref: Tensor, dim: int) -> Tensor:
    size = [1] * ref.dim()
    size[dim] = -1
    return src.view(size).expand_as(ref)
