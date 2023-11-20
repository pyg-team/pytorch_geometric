import errno
import os
import os.path as osp

from torch_geometric.data.fs_utils import fs_isdir, fs_mkdirs, fs_normpath


def makedirs(path: str):
    r"""Recursively creates a directory.

    Args:
        path (str): The path to create.
    """
    try:
        fs_mkdirs(osp.expanduser(fs_normpath(path)))
    except FileExistsError as e:
        if not fs_isdir(path):
            raise e
