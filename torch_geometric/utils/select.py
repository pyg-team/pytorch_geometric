from typing import Any, List, Union

import torch
from torch import Tensor

from torch_geometric.utils.mask import mask_select
from torch_geometric.utils.sparse import is_torch_sparse_tensor


def select(src: Union[Tensor, List[Any]], index_or_mask: Tensor,
           dim: int) -> Union[Tensor, List[Any]]:
    r"""Selects the input tensor or input list according to a given index or
    mask vector.

    Args:
        src (torch.Tensor or list): The input tensor or list.
        index_or_mask (torch.Tensor): The index or mask vector.
        dim (int): The dimension along which to select.
    """
    if isinstance(src, Tensor):
        if index_or_mask.dtype == torch.bool:
            return mask_select(src, dim, index_or_mask)
        return src.index_select(dim, index_or_mask)

    if isinstance(src, (tuple, list)):
        if dim != 0:
            raise ValueError("Cannot select along dimension other than 0")
        if index_or_mask.dtype == torch.bool:
            return [src[i] for i, m in enumerate(index_or_mask) if m]
        return [src[i] for i in index_or_mask]

    raise ValueError(f"Encountered invalid input type (got '{type(src)}')")


def narrow(src: Union[Tensor, List[Any]], dim: int, start: int,
           length: int) -> Union[Tensor, List[Any]]:
    r"""Narrows the input tensor or input list to the specified range.

    Args:
        src (torch.Tensor or list): The input tensor or list.
        dim (int): The dimension along which to narrow.
        start (int): The starting dimension.
        length (int): The distance to the ending dimension.
    """
    if is_torch_sparse_tensor(src):
        # TODO Sparse tensors in `torch.sparse` do not yet support `narrow`.
        index = torch.arange(start, start + length, device=src.device)
        return src.index_select(dim, index)

    if isinstance(src, Tensor):
        return src.narrow(dim, start, length)

    if isinstance(src, list):
        if dim != 0:
            raise ValueError("Cannot narrow along dimension other than 0")
        return src[start:start + length]

    raise ValueError(f"Encountered invalid input type (got '{type(src)}')")
