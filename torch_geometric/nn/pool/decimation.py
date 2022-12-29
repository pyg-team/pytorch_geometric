from typing import Tuple, Union

import torch
from torch import LongTensor, Tensor


def decimation_indices(
    ptr: LongTensor,
    decimation_factor: Union[int, float],
) -> Tuple[Tensor, LongTensor]:
    """Gets indices which downsample each point cloud by a decimation factor.

    Decimation happens separately for each cloud to prevent emptying smaller
    point clouds. Empty clouds are prevented: clouds will have a least
    one node after decimation.

    Args:
        ptr (LongTensor): The indices of samples in the batch.
        decimation_factor (int or float): The value to divide number of nodes
            with. Should be higher than (or equal to) :obj:`1` for
            downsampling.

    :rtype: (:class:`LongTensor`, :class:`LongTensor`): The indices and
        updated :obj:`ptr` after downsampling.
    """
    if decimation_factor < 1:
        raise ValueError(
            f"The argument `decimation_factor` should be higher than (or "
            f"equal to) 1 for downsampling. (got {decimation_factor})")

    batch_size = ptr.size(0) - 1
    count = ptr[1:] - ptr[:-1]
    decim_count = torch.div(count, decimation_factor, rounding_mode='floor')
    decim_count.clamp_(min=1)  # Prevent empty examples.

    decim_indices = [
        ptr[i] + torch.randperm(count[i], device=ptr.device)[:decim_count[i]]
        for i in range(batch_size)
    ]
    decim_indices = torch.cat(decim_indices, dim=0)

    # Get updated ptr (e.g., for future decimations):
    decim_ptr = torch.cat([decim_count.new_zeros(1), decim_count.cumsum(0)])

    return decim_indices, decim_ptr
