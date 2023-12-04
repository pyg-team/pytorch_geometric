from torch_geometric.deprecation import deprecated
from torch_geometric.io import fs


@deprecated("use 'os.makedirs(path, exist_ok=True)' instead")
def makedirs(path: str):
    r"""Recursively creates a directory.

    .. warning::

        :meth:`makedirs` is deprecated and will be removed soon.
        Please use :obj:`os.makedirs(path, exist_ok=True)` instead.

    Args:
        path (str): The path to create.
    """
    fs.makedirs(path, exist_ok=True)
