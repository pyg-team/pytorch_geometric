from typing import Tuple
import torch
from torch import Tensor, LongTensor
from numbers import Number


def decimation_indices(
    ptr: LongTensor, decimation_factor: Number
) -> Tuple[Tensor, Tensor]:
    """Get indices which downsample each point cloud by a decimation factor.

    Decimation happens separately for each cloud to prevent emptying smaller
    point clouds. Empty clouds are prevented: clouds will have a least
    one node after decimation.

    Args:
        ptr (LongTensor): indices of each sample's first point in the batch.
        decimation_factor (Number): value to divide number of nodes with.
            Should be higher than 1 for downsampling.

    :rtype: (:class:`Tensor`, :class:`LongTensor`): indices for downsampling
        and resulting updated ptr.

    """
    if decimation_factor < 1:
        raise ValueError(
            "Argument `decimation_factor` should be higher than (or equal to) 1 for "
            f"downsampling. (Current value: {decimation_factor})"
        )

    batch_size = ptr.size(0) - 1
    bincount = ptr[1:] - ptr[:-1]
    decimated_bincount = torch.div(
        bincount, decimation_factor, rounding_mode="floor"
    )
    # Decimation should not empty clouds completely.
    decimated_bincount = torch.max(
        torch.ones_like(decimated_bincount), decimated_bincount
    )
    idx_decim = torch.cat(
        [
            (
                ptr[i]
                + torch.randperm(bincount[i], device=ptr.device)[
                    : decimated_bincount[i]
                ]
            )
            for i in range(batch_size)
        ],
        dim=0,
    )
    # Get updated ptr (e.g. for future decimations)
    ptr_decim = ptr.clone()
    for i in range(batch_size):
        ptr_decim[i + 1] = ptr_decim[i] + decimated_bincount[i]

    return idx_decim, ptr_decim
