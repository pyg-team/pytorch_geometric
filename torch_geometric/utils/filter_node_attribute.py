from typing import List, Union

import torch
from torch import Tensor

from .mask import mask_select


def filter_node_attribute(value: Union[Tensor, List], dim: int,
                          subset: Tensor) -> Union[Tensor, List]:
    r"""Filters a node attribute according to the given subset.
    The value can be a tensor or list. For lists, the :obj:`dim` is ignored.
    Args:
        value (Tensor or List): The input tensor or list.
        dim (int): The dimension along which to filter.
        subset (LongTensor or BoolTensor): The nodes to keep.
    """
    if isinstance(value, Tensor):
        if subset.dtype == torch.bool:
            return mask_select(value, dim, subset)
        else:
            return value.index_select(dim, subset)
    elif isinstance(value, list):
        if subset.dtype == torch.bool:
            return [value[i] for i, x in enumerate(subset) if x]
        else:
            return [value[i] for i in subset]

    raise ValueError(f"Encountered invalid feature tensor type "
                     f"(got '{type(value)}')")


def narrow_node_attribute(value: Union[Tensor, List], dim: int, start: int,
                          length: int) -> Union[Tensor, List]:
    r"""Narrows a node attribute to the specified range.
    For tensors, the dimension :obj:`dim` is input from
    :obj:`start` to :obj:`start` + :obj:`length`.
    For lists, the :obj:`dim` is ignored.
    Args:
        value (Tensor or List): The input tensor or list.
        dim (int): The dimension along which to narrow.
        start (int): The starting dimension.
        length (int): The distance to the ending dimension.
    """
    if isinstance(value, Tensor):
        return value.narrow(dim, start, length)
    elif isinstance(value, list):
        return value[start:start + length]

    raise ValueError(f"Encountered invalid feature tensor type "
                     f"(got '{type(value)}')")
