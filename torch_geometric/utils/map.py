from typing import Tuple

import torch
from torch import Tensor


def map_index(src: Tensor, index: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Maps indices in :obj:`src` to the positional value of their
    corresponding occurence in :obj:`index`.

    Args:
        src (torch.Tensor): The source tensor to map.
        index (torch.Tensor): The index tensor that denotes the new mapping.

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
    """
    import pandas as pd

    assert src.dim() == 1 and index.dim() == 1
    assert not src.is_floating_point()
    assert not index.is_floating_point()

    arange = pd.RangeIndex(0, index.size(0))
    df = pd.DataFrame(index=index.detach().cpu().numpy(), data={'out': arange})
    ser = pd.Series(src.detach().cpu(), name='key')
    result = df.merge(ser, how='right', left_index=True, right_on='key')
    out = torch.from_numpy(result['out'].values).to(index.device)

    if out.is_floating_point():
        mask = torch.isnan(out).logical_not_()
        out = out[mask].to(index.dtype)
        return out, mask

    out = out.to(index.dtype)
    mask = torch.ones_like(out, dtype=torch.bool)
    return out, mask
