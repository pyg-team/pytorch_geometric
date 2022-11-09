from typing import Any

from torch import Tensor


def is_torch_sparse_tensor(src: Any) -> bool:
    """Returns :obj:`True` if the input :obj:`x` is a PyTorch
    :obj:`SparseTensor` (in any sparse format).

    Args:
        src (Any): The input object to be checked.
    """
    return isinstance(src, Tensor) and src.is_sparse
