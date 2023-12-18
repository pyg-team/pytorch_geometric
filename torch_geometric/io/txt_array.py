from typing import List, Optional

import fsspec
import torch
from torch import Tensor


def parse_txt_array(
    src: List[str],
    sep: Optional[str] = None,
    start: int = 0,
    end: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    empty = torch.empty(0, dtype=dtype)
    to_number = float if empty.is_floating_point() else int

    return torch.tensor([[to_number(x) for x in line.split(sep)[start:end]]
                         for line in src], dtype=dtype).squeeze()


def read_txt_array(
    path: str,
    sep: Optional[str] = None,
    start: int = 0,
    end: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    with fsspec.open(path, 'r') as f:
        src = f.read().split('\n')[:-1]
    return parse_txt_array(src, sep, start, end, dtype, device)
