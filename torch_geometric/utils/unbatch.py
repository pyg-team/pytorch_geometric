import torch

from torch_geometric.utils import degree


def unbatch(src: torch.tensor, index: torch.tensor, dim: int = -1) -> list:
    r"""Unbatches batched data from DataLoader.

    Args:
        src (Tensor): source tensor.
        index (LongTensor): index tensor.
        dim (int): dimension along which to split the tensor.

    :rtype: :class:`list`
    """
    split_sizes = list(map(int, degree(index).tolist()))
    unbatched = src.split(split_sizes, dim)
    return unbatched
