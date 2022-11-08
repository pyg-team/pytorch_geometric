from typing import Any

from torch import Tensor


def is_torch_sparse_tensor(x: Any) -> bool:
    """Returns :obj:`True` if the input :obj:`x` is
    PyTorch SparseTensor (COO or CSR format).

    Args:
        x (Any): The input object to be checked.
    """
    return isinstance(x, Tensor) and x.is_sparse
