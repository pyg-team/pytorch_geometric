from typing import Optional

import torch
from torch import Tensor

from torch_geometric.typing import TensorFrame


def mask_select(src: Tensor, dim: int, mask: Tensor) -> Tensor:
    r"""Returns a new tensor which masks the :obj:`src` tensor along the
    dimension :obj:`dim` according to the boolean mask :obj:`mask`.

    Args:
        src (torch.Tensor): The input tensor.
        dim (int): The dimension in which to mask.
        mask (torch.BoolTensor): The 1-D tensor containing the binary mask to
            index with.
    """
    assert mask.dim() == 1

    if not torch.jit.is_scripting():
        if isinstance(src, TensorFrame):
            assert dim == 0 and src.num_rows == mask.numel()
            return src[mask]

    assert src.size(dim) == mask.numel()
    dim = dim + src.dim() if dim < 0 else dim
    assert dim >= 0 and dim < src.dim()

    size = [1] * src.dim()
    size[dim] = mask.numel()

    out = src.masked_select(mask.view(size))

    size = list(src.size())
    size[dim] = -1

    return out.view(size)


def index_to_mask(index: Tensor, size: Optional[int] = None) -> Tensor:
    r"""Converts indices to a mask representation.

    Args:
        idx (Tensor): The indices.
        size (int, optional). The size of the mask. If set to :obj:`None`, a
            minimal sized output mask is returned.

    Example:

        >>> index = torch.tensor([1, 3, 5])
        >>> index_to_mask(index)
        tensor([False,  True, False,  True, False,  True])

        >>> index_to_mask(index, size=7)
        tensor([False,  True, False,  True, False,  True, False])
    """
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask


def mask_to_index(mask: Tensor) -> Tensor:
    r"""Converts a mask to an index representation.

    Args:
        mask (Tensor): The mask.

    Example:

        >>> mask = torch.tensor([False, True, False])
        >>> mask_to_index(mask)
        tensor([1])
    """
    return mask.nonzero(as_tuple=False).view(-1)
