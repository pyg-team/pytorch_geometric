from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.utils import scatter


def to_nested_tensor(
    x: Tensor,
    batch: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    batch_size: Optional[int] = None,
) -> Tensor:
    r"""Given a contiguous batch of tensors
    :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times *}`
    (with :math:`N_i` indicating the number of elements in example :math:`i`),
    creates a `nested PyTorch tensor
    <https://pytorch.org/docs/stable/nested.html>`__.
    Reverse operation of :meth:`from_nested_tensor`.

    Args:
        x (torch.Tensor): The input tensor
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times *}`.
        batch (torch.Tensor, optional): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            element to a specific example. Must be ordered.
            (default: :obj:`None`)
        ptr (torch.Tensor, optional): Alternative representation of
            :obj:`batch` in compressed format. (default: :obj:`None`)
        batch_size (int, optional): The batch size :math:`B`.
            (default: :obj:`None`)
    """
    if ptr is not None:
        offsets = ptr[1:] - ptr[:-1]
        sizes = offsets.tolist()
        xs = list(torch.split(x, sizes, dim=0))
    elif batch is not None:
        offsets = scatter(torch.ones_like(batch), batch, dim_size=batch_size)
        sizes = offsets.tolist()
        xs = list(torch.split(x, sizes, dim=0))
    else:
        xs = [x]

    # This currently copies the data, although `x` is already contiguous.
    # Sadly, there does not exist any (public) API to prevent this :(
    return torch.nested.as_nested_tensor(xs)


def from_nested_tensor(
    x: Tensor,
    return_batch: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    r"""Given a `nested PyTorch tensor
    <https://pytorch.org/docs/stable/nested.html>`__, creates a contiguous
    batch of tensors
    :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times *}`, and
    optionally a batch vector which assigns each element to a specific example.
    Reverse operation of :meth:`to_nested_tensor`.

    Args:
        x (torch.Tensor): The nested input tensor. The size of nested tensors
            need to match except for the first dimension.
        return_batch (bool, optional): If set to :obj:`True`, will also return
            the batch vector :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`.
            (default: :obj:`False`)
    """
    if not x.is_nested:
        raise ValueError("Input tensor in 'from_nested_tensor' is not nested")

    sizes = x._nested_tensor_size()

    for dim, (a, b) in enumerate(zip(sizes[0, 1:], sizes.t()[1:])):
        if not torch.equal(a.expand_as(b), b):
            raise ValueError(f"Not all nested tensors have the same size "
                             f"in dimension {dim + 1} "
                             f"(expected size {a.item()} for all tensors)")

    out = x.contiguous().values()
    out = out.view(-1, *sizes[0, 1:].tolist())

    if not return_batch:
        return out

    batch = torch.arange(x.size(0), device=x.device)
    batch = batch.repeat_interleave(sizes[:, 0].to(batch.device))

    return out, batch
