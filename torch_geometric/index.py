from typing import Optional

import torch
from torch import Tensor


def ptr2index(ptr: Tensor, output_size: Optional[int] = None) -> Tensor:
    index = torch.arange(ptr.numel() - 1, dtype=ptr.dtype, device=ptr.device)
    return index.repeat_interleave(ptr.diff(), output_size=output_size)


def index2ptr(index: Tensor, size: Optional[int] = None) -> Tensor:
    if size is None:
        size = int(index.max()) + 1 if index.numel() > 0 else 0

    return torch._convert_indices_from_coo_to_csr(
        index, size, out_int32=index.dtype != torch.int64)
