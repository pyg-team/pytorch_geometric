import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric import is_compiling
from torch_geometric.typing import torch_scatter


def segment(src: Tensor, ptr: Tensor, reduce: str = 'sum') -> Tensor:
    r"""Reduces all values in the first dimension of the :obj:`src` tensor
    within the ranges specified in the :obj:`ptr`. See the `documentation
    <https://pytorch-scatter.readthedocs.io/en/latest/functions/
    segment_csr.html>`__ of the :obj:`torch_scatter` package for more
    information.

    Args:
        src (torch.Tensor): The source tensor.
        ptr (torch.Tensor): A monotonically increasing pointer tensor that
            refers to the boundaries of segments such that :obj:`ptr[0] = 0`
            and :obj:`ptr[-1] = src.size(0)`.
        reduce (str, optional): The reduce operation (:obj:`"sum"`,
            :obj:`"mean"`, :obj:`"min"` or :obj:`"max"`).
            (default: :obj:`"sum"`)
    """
    if not torch_geometric.typing.WITH_TORCH_SCATTER or is_compiling():
        return _torch_segment(src, ptr, reduce)

    if (ptr.dim() == 1 and torch_geometric.typing.WITH_PT20 and src.is_cuda
            and reduce == 'mean'):
        return _torch_segment(src, ptr, reduce)

    return torch_scatter.segment_csr(src, ptr, reduce=reduce)


def _torch_segment(src: Tensor, ptr: Tensor, reduce: str = 'sum') -> Tensor:
    if not torch_geometric.typing.WITH_PT20:
        raise ImportError("'segment' requires the 'torch-scatter' package")
    if ptr.dim() > 1:
        raise ImportError("'segment' in an arbitrary dimension "
                          "requires the 'torch-scatter' package")

    if reduce == 'min' or reduce == 'max':
        reduce = f'a{reduce}'  # `amin` or `amax`
    initial = 0 if reduce == 'mean' else None
    out = torch._segment_reduce(src, reduce, offsets=ptr, initial=initial)
    if reduce == 'amin' or reduce == 'amax':
        out = torch.where(out.isinf(), 0, out)
    return out
