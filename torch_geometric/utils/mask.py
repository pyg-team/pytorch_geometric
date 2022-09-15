from typing import Optional

import torch
from torch import Tensor


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
