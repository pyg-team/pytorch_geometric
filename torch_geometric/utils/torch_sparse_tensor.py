from typing import Any


def is_torch_sparse_tensor(x: Any) -> bool:
    """Returns :obj:`True` if the input :obj:`x` is
    PyTorch SparseTensor (COO or CSR format).

    Args:
        x (Any): The input object to be checked.
    """
    return getattr(x, 'is_sparse', False)
