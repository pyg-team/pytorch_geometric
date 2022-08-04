from typing import Optional

import torch
from torch import Tensor

from .broadcast import broadcast


def prepare_out(src: Tensor, index: Tensor, dim: int = -1,
                dim_size: Optional[int] = None, reduce: str = 'sum') -> Tensor:
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1

    return torch.zeros(size, dtype=src.dtype, device=src.device)


def scatter_sum(src: Tensor, index: Tensor, dim: int = -1,
                out: Optional[Tensor] = None,
                dim_size: Optional[int] = None) -> Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        return prepare_out(src, index, dim,
                           dim_size).scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def scatter_mean(src: Tensor, index: Tensor, dim: int = -1,
                 out: Optional[Tensor] = None,
                 dim_size: Optional[int] = None) -> Tensor:
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode='floor')

    return out


def scatter_reduce(src: Tensor, index: Tensor, dim: int = -1,
                   out: Optional[Tensor] = None,
                   dim_size: Optional[int] = None,
                   reduce: str = 'sum') -> Tensor:
    index = broadcast(index, src, dim)
    include_self = out is not None
    if out is None:
        out = prepare_out(src, index, dim, dim_size)
        return out.scatter_reduce_(dim, index, src, reduce,
                                   include_self=include_self)
    else:
        return out.scatter_reduce_(dim, index, src, reduce,
                                   include_self=include_self)


def scatter(src: Tensor, index: Tensor, dim: int = -1,
            out: Optional[Tensor] = None, dim_size: Optional[int] = None,
            reduce: str = 'sum') -> Tensor:
    r"""Performs reduction of all values from the :attr:`src` tensor into
    :attr:`out` at the indices specified in the :attr:`index` tensor along
    a given axis :attr:`dim`.
    If :obj:`torch` version :obj:`>=1.12.0` this operation acts like a thin
    wrapper around :meth:`torch.scatter_reduce` which makes its API aligned
    with :meth:`torch_scatter.reduce` - marks :attr:`out` as optional input
    and allows :attr:`src` and :attr:`index` to broadcast. Otherwise,
    execution falls back to :meth:`torch_scatter.reduce` implementation.

    .. note::
        This operations performs with much better performance when
        :obj:`torch` version :obj:`>=1.12.0`.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim (int, optional): The axis along which to index.
            (default: :obj:`-1`)
        out (Tensor, optional): The destination tensor.
        dim_size (int, optional): If :attr:`out` is not given, automatically
            create output with size :attr:`dim_size` at dimension :attr:`dim`.
            If :attr:`dim_size` is not given, a minimal sized output tensor
            according to :obj:`index.max() + 1` is returned.
        reduce (str, optional): The reduce operation (:obj:`"sum"`,
            :obj:`"mul"`, :obj:`"mean"`, :obj:`"min"` or :obj:`"max"`).
            (default: :obj:`"sum"`)

    :rtype: :class:`Tensor`

    .. code-block:: python

        from torch_geometric.utils import scatter

        src = torch.randn(10, 6, 64)
        index = torch.tensor([0, 1, 0, 1, 2, 1])

        # Broadcasting in the first and last dim.
        out = scatter(src, index, dim=1, reduce="sum")

        print(out.size())
        >>> torch.Size([10, 3, 64])
    """
    if reduce == 'sum' or reduce == 'add':
        return scatter_sum(src, index, dim, out, dim_size)
    elif reduce == 'mean':
        return scatter_mean(src, index, dim, out, dim_size)
    elif reduce == 'mul':
        return scatter_reduce(src, index, dim, out, dim_size, 'prod')
    elif reduce == 'min':
        return scatter_reduce(src, index, dim, out, dim_size, 'amin')
    elif reduce == 'max':
        return scatter_reduce(src, index, dim, out, dim_size, 'amax')
    else:
        raise ValueError
