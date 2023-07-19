from typing import List

import numpy as np
import torch
from torch import Tensor

import torch_geometric.typing


def lexsort(
    keys: List[Tensor],
    dim: int = -1,
    descending: bool = False,
) -> Tensor:
    r"""Performs an indirect stable sort using a sequence of keys.

    Given multiple sorting keys, returns an array of integer indices that
    describe their sort order.
    The last key in the sequence is used for the primary sort order, the
    second-to-last key for the secondary sort order, and so on.

    Args:
        keys ([torch.Tensor]): The :math:`k` different columns to be sorted.
            The last key is the primary sort key.
        dim (int, optional): The dimension to sort along. (default: :obj:`-1`)
        descending (bool, optional): Controls the sorting order (ascending or
            descending). (default: :obj:`False`)
    """
    assert len(keys) >= 1

    if not torch_geometric.typing.WITH_PT113:
        keys = [k.neg() for k in keys] if descending else keys
        out = np.lexsort([k.detach().cpu().numpy() for k in keys], axis=dim)
        return torch.from_numpy(out).to(keys[0].device)

    kwargs = dict(dim=dim, descending=descending, stable=True)
    out = keys[0].argsort(**kwargs)
    for k in keys[1:]:
        out = out.gather(dim, k.gather(dim, out).argsort(**kwargs))

    return out
