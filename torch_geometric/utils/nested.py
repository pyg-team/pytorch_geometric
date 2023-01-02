from typing import Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.utils import scatter


def to_nested_tensor(
    x: Tensor,
    batch: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    if ptr is not None:
        sizes = ptr[1:] - ptr[:-1]
        xs = list(torch.split(x, sizes.tolist(), dim=0))
    elif batch is not None:
        sizes = scatter(torch.ones_like(batch), batch, dim_size=batch_size)
        xs = list(torch.split(x, sizes.tolist(), dim=0))
    else:
        xs = [x]

    # This currently copies the data, although `x` is already contiguous.
    # Sadly, there exists currently no way to prevent this :(
    return torch.nested.as_nested_tensor(xs)


def from_nested_tensor(x: Tensor) -> Tuple[Tensor, Tensor]:
    if not x.is_nested:
        raise ValueError
    # print("ääääääääääääääääää")
    # print(x.is_nested)
    # print(x.__class__.__dict__.keys())
    # # print(x.get_nested_size_tensor)
    # print(x.layout)
    # print(x.size(0))
    print(x._nested_tensor_size())
    sizes = x._nested_tensor_size()[:, 0]
    print(sizes)

    # print(x.size(1))
    # print(x.size(1))
    # print(x.nested_dim)
    # x = x.contiguous()
    # TODO This will lose grad information!!!
    torch.tensor(x.contiguous().storage())
    pass
