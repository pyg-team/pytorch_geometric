import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.utils.dlpack import from_dlpack


def map_index(
    src: Tensor,
    index: Tensor,
    max_index: Optional[int] = None,
    inclusive: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""Maps indices in :obj:`src` to the positional value of their
    corresponding occurence in :obj:`index`.
    Indices must be strictly positive.

    Args:
        src (torch.Tensor): The source tensor to map.
        index (torch.Tensor): The index tensor that denotes the new mapping.
        max_index (int, optional): The maximum index value.
            (default :obj:`None`)
        inclusive (bool, optional): If set to :obj:`True`, it is assumed that
            every entry in :obj:`src` has a valid entry in :obj:`index`.
            Can speed-up computation. (default: :obj:`False`)

    :rtype: (:class:`torch.Tensor`, :class:`torch.BoolTensor`)

    Examples:
        >>> src = torch.tensor([2, 0, 1, 0, 3])
        >>> index = torch.tensor([3, 2, 0, 1])

        >>> map_index(src, index)
        (tensor([1, 2, 3, 2, 0]), tensor([True, True, True, True, True]))

        >>> src = torch.tensor([2, 0, 1, 0, 3])
        >>> index = torch.tensor([3, 2, 0])

        >>> map_index(src, index)
        (tensor([1, 2, 2, 0]), tensor([True, True, False, True, True]))

    .. note::

        If inputs are on GPU and :obj:`cudf` is available, consider using RMM
        for significant speed boosts.
        Proceed with caution as RMM may conflict with other allocators or
        fragments.

        .. code-block:: python

            import rmm
            rmm.reinitialize(pool_allocator=True)
            torch.cuda.memory.change_current_allocator(rmm.rmm_torch_allocator)
    """
    if src.is_floating_point():
        raise ValueError(f"Expected 'src' to be an index (got '{src.dtype}')")
    if index.is_floating_point():
        raise ValueError(f"Expected 'index' to be an index (got "
                         f"'{index.dtype}')")
    if src.device != index.device:
        raise ValueError(f"Both 'src' and 'index' must be on the same device "
                         f"(got '{src.device}' and '{index.device}')")

    if max_index is None:
        max_index = max(src.max(), index.max())

    # If the `max_index` is in a reasonable range, we can accelerate this
    # operation by creating a helper vector to perform the mapping.
    # NOTE This will potentially consumes a large chunk of memory
    # (max_index=10 million => ~75MB), so we cap it at a reasonable size:
    THRESHOLD = 40_000_000 if src.is_cuda else 10_000_000
    if max_index <= THRESHOLD:
        if inclusive:
            assoc = src.new_empty(max_index + 1)
        else:
            assoc = src.new_full((max_index + 1, ), -1)
        assoc[index] = torch.arange(index.numel(), dtype=src.dtype,
                                    device=src.device)
        out = assoc[src]

        if inclusive:
            return out, None
        else:
            mask = out != -1
            return out[mask], mask

    WITH_CUDF = False
    if src.is_cuda:
        try:
            import cudf
            WITH_CUDF = True
        except ImportError:
            import pandas as pd
            warnings.warn("Using CPU-based processing within 'map_index' "
                          "which may cause slowdowns and device "
                          "synchronization. Consider installing 'cudf' to "
                          "accelerate computation")
    else:
        import pandas as pd

    if not WITH_CUDF:
        left_ser = pd.Series(src.cpu().numpy(), name='left_ser')
        right_ser = pd.Series(
            index=index.cpu().numpy(),
            data=pd.RangeIndex(0, index.size(0)),
            name='right_ser',
        )

        result = pd.merge(left_ser, right_ser, how='left', left_on='left_ser',
                          right_index=True)

        out = torch.from_numpy(result['right_ser'].values).to(index.device)

        if out.is_floating_point() and inclusive:
            raise ValueError("Found invalid entries in 'src' that do not have "
                             "a corresponding entry in 'index'. Set "
                             "`inclusive=False` to ignore these entries.")

        if out.is_floating_point():
            mask = torch.isnan(out).logical_not_()
            out = out[mask].to(index.dtype)
            return out, mask

        if inclusive:
            return out, None
        else:
            mask = out != -1
            return out[mask], mask

    else:
        left_ser = cudf.Series(src, name='left_ser')
        right_ser = cudf.Series(
            index=index,
            data=cudf.RangeIndex(0, index.size(0)),
            name='right_ser',
        )

        result = cudf.merge(left_ser, right_ser, how='left',
                            left_on='left_ser', right_index=True, sort=True)

        if inclusive:
            try:
                out = from_dlpack(result['right_ser'].to_dlpack())
            except ValueError:
                raise ValueError("Found invalid entries in 'src' that do not "
                                 "have a corresponding entry in 'index'. Set "
                                 "`inclusive=False` to ignore these entries.")
        else:
            out = from_dlpack(result['right_ser'].fillna(-1).to_dlpack())

        out = out[src.argsort().argsort()]  # Restore original order.

        if inclusive:
            return out, None
        else:
            mask = out != -1
            return out[mask], mask
