from typing import Optional

import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric import warnings
from torch_geometric.typing import torch_scatter
from torch_geometric.utils.functions import cumsum

if torch_geometric.typing.WITH_PT112:  # pragma: no cover

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
        if isinstance(index, Tensor) and index.dim() != 1:
            raise ValueError(f"The `index` argument must be one-dimensional "
                             f"(got {index.dim()} dimensions)")

        dim = src.dim() + dim if dim < 0 else dim

        if isinstance(src, Tensor) and (dim < 0 or dim >= src.dim()):
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

        size = src.size()[:dim] + (dim_size, ) + src.size()[dim + 1:]

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

            size = src.size()[:dim] + (dim_size, ) + src.size()[dim + 1:]

            index = broadcast(index, src, dim)
            return src.new_zeros(size).scatter_(dim, index, src)

        if not torch_geometric.typing.WITH_TORCH_SCATTER:
            raise ImportError("'scatter' requires the 'torch-scatter' package")
        return torch_scatter.scatter(src, index, dim, dim_size=dim_size,
                                     reduce=reduce)


def broadcast(src: Tensor, ref: Tensor, dim: int) -> Tensor:
    size = ((1, ) * dim) + (-1, ) + ((1, ) * (ref.dim() - dim - 1))
    return src.view(size).expand_as(ref)


def scatter_argmax(src: Tensor, index: Tensor, dim: int = 0,
                   dim_size: Optional[int] = None) -> Tensor:

    if torch_geometric.typing.WITH_TORCH_SCATTER:
        out = torch_scatter.scatter_max(src, index, dim=dim, dim_size=dim_size)
        return out[1]

    # Only implemented under certain conditions for now :(
    assert src.dim() == 1 and index.dim() == 1
    assert dim == 0 or dim == -1
    assert src.numel() == index.numel()

    if dim_size is None:
        dim_size = index.max() + 1 if index.numel() > 0 else 0

    if torch_geometric.typing.WITH_PT112:
        res = src.new_empty(dim_size)
        res.scatter_reduce_(0, index, src.detach(), reduce='amax',
                            include_self=False)
    elif torch_geometric.typing.WITH_PT111:
        res = torch.scatter_reduce(src.detach(), 0, index, reduce='amax',
                                   output_size=dim_size)
    else:
        raise ValueError("'scatter_argmax' requires PyTorch >= 1.11")

    out = index.new_full((dim_size, ), fill_value=dim_size - 1)
    nonzero = (src == res[index]).nonzero().view(-1)
    out[index[nonzero]] = nonzero

    return out


def group_argsort(
    src: Tensor,
    index: Tensor,
    dim: int = 0,
    num_groups: Optional[int] = None,
    descending: bool = False,
    return_consecutive: bool = False,
    stable: bool = False,
) -> Tensor:
    r"""Returns the indices that sort the tensor :obj:`src` along a given
    dimension in ascending order by value.
    In contrast to :meth:`torch.argsort`, sorting is performed in groups
    according to the values in :obj:`index`.

    Args:
        src (torch.Tensor): The source tensor.
        index (torch.Tensor): The index tensor.
        dim (int, optional): The dimension along which to index.
            (default: :obj:`0`)
        num_groups (int, optional): The number of groups.
            (default: :obj:`None`)
        descending (bool, optional): Controls the sorting order (ascending or
            descending). (default: :obj:`False`)
        return_consecutive (bool, optional): If set to :obj:`True`, will not
            offset the output to start from :obj:`0` for each group.
            (default: :obj:`False`)
        stable (bool, optional): Controls the relative order of equivalent
            elements. (default: :obj:`False`)
    """
    # Only implemented under certain conditions for now :(
    assert src.dim() == 1 and index.dim() == 1
    assert dim == 0 or dim == -1
    assert src.numel() == index.numel() and src.numel() > 0

    # Normalize `src` to range [0, 1]:
    src = src - src.min()
    src = src / src.max()

    # Compute `grouped_argsort`:
    src = src - 2 * index if descending else src + 2 * index
    if torch_geometric.typing.WITH_PT113:
        perm = src.argsort(descending=descending, stable=stable)
    else:
        perm = src.argsort(descending=descending)
        if stable:
            warnings.warn("Ignoring option `stable=True` in 'group_argsort' "
                          "since it requires PyTorch >= 1.13.0")
    out = torch.empty_like(index)
    out[perm] = torch.arange(index.numel(), device=index.device)

    if return_consecutive:
        return out

    # Compute cumulative sum of number of entries with the same index:
    count = scatter(torch.ones_like(index), index, dim=dim,
                    dim_size=num_groups, reduce='sum')
    ptr = cumsum(count)

    return out - ptr[index]
