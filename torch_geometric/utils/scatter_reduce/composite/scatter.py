from typing import Optional

from torch import Tensor

from .logsumexp import scatter_logsumexp
from .softmax import scatter_log_softmax, scatter_softmax
from .std import scatter_std


def scatter_composite(src: Tensor, index: Tensor, dim: int = -1,
                      out: Optional[Tensor] = None,
                      dim_size: Optional[int] = None, unbiased: bool = True,
                      reduce: str = 'std') -> Tensor:
    r"""Performs reduction for complex reduction algorithms, using
    :meth:`~torch_geometric.utils.scatter` operation under the hood.

    .. warning::
        Reduction for :obj:`"logsumexp"`, :obj:`"logsoftmax"` and
        :obj:`"softmax"` algorithms can only be computed over tensor with
        floating data types.

    .. note::
        This operations performs with much better performance when
        :obj:`torch` version :obj:`>=1.12.0`.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim (int, optional): The axis along which to index.
            (default: :obj:`-1`)
        out (Tensor, optional): The destination tensor. Ignored for
            :attr:`reduce` equal to :obj:`"logsoftmax"` or :obj:`"softmax"`.
        dim_size (int, optional): If :attr:`out` is not given, automatically
            create output with size :attr:`dim_size` at dimension :attr:`dim`.
            If :attr:`dim_size` is not given, a minimal sized output tensor
            according to :obj:`index.max() + 1` is returned.
        unbiased (bool, optional): Consumable only by :obj:`"std"` reduction.
            If set to :obj:`True` standard deviation is computed for sample of
            population, otherwise for entire population. (default: :obj:`True`)
        reduce (str, optional): The reduce operation (:obj:`"std"`,
            :obj:`"logsumexp"`, :obj:`"logsoftmax"` or :obj:`"softmax"`).
            (default: :obj:`"std"`)

    :rtype: :class:`Tensor`
    """
    if reduce == 'std':
        return scatter_std(src, index, dim, out, dim_size, unbiased)
    if reduce == 'logsumexp':
        return scatter_logsumexp(src, index, dim, out, dim_size)
    elif reduce == 'logsoftmax':
        return scatter_log_softmax(src, index, dim, dim_size)
    elif reduce == 'softmax':
        return scatter_softmax(src, index, dim, dim_size)
    else:
        raise ValueError
