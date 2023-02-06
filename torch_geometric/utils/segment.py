from torch import Tensor

import torch_geometric.typing
from torch_geometric.typing import torch_scatter


def segment(src: Tensor, ptr: Tensor, reduce: str = 'sum') -> Tensor:
    r"""Reduces all values in the first dimension of the :obj:`src` tensor
    within the ranges specified in the :obj:`ptr`. See the `documentation
    <https://pytorch-scatter.readthedocs.io/en/latest/functions/
    segment_csr.html>`__ of :obj:`torch-scatter` for more information.

    Args:
        src (torch.Tensor): The source tensor.
        ptr (torch.Tensor): A monotonically increasing pointer tensor that
            refers to the boundaries of segments such that :obj:`ptr[0] = 0`
            and :obj:`ptr[-1] = src.size(0)`.
        reduce (str, optional): The reduce operation (:obj:`"sum"`,
            :obj:`"mean"`, :obj:`"mul"`, :obj:`"min"` or :obj:`"max"`).
            (default: :obj:`"sum"`)
    """
    if not torch_geometric.typing.WITH_TORCH_SCATTER:
        raise ImportError("'segment' requires the 'torch-scatter' package")
    return torch_scatter.segment_csr(src, ptr, reduce=reduce)
