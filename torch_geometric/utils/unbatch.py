import torch
from torch import Tensor

from torch_geometric.utils import degree


def unbatch(src: Tensor, batch: Tensor, dim: int = 0) -> List[Tensor]:
    r"""Unbatches batched data from DataLoader.

    Args:
        src (Tensor): source tensor.
        batch (LongTensor): index tensor.
        dim (int): dimension along which to split the tensor.

    :rtype: :class:`list`
    """
    split_sizes = degree(batch, dtype=torch.long).tolist()
    unbatched = list(src.split(split_sizes, dim))
    return unbatched
