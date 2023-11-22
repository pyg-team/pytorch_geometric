import os
import os.path as osp

from torch_geometric.deprecation import deprecated


@deprecated("use 'os.makedirs(path, exist_ok=True)' instead")
def makedirs(path: str):
    r"""Recursively creates a directory.

    .. warning::

        :meth:`makedirs` is deprecated and will be removed soon.
        Please use :obj:`os.makedirs(path, exist_ok=True)` instead.

    Args:
        path (str): The path to create.
    """
    os.makedirs(osp.expanduser(osp.normpath(path)), exist_ok=True)
