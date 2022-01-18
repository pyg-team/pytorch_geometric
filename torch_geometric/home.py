from typing import Optional

import os
import os.path as osp

ENV_PYG_HOME = 'PYG_HOME'
DEFAULT_CACHE_DIR = osp.join('~', '.cache', 'pyg')

_home_dir: Optional[str] = None


def get_home_dir() -> str:
    r"""Get the cache directory used for storing all PyG-related data.

    If :meth:`set_home_dir` is not called, the path is given by the environment
    variable :obj:`$PYG_HOME` which defaults to :obj:`"~/.cache/pyg"`.
    """
    if _home_dir is not None:
        return _home_dir

    home_dir = os.getenv(ENV_PYG_HOME, DEFAULT_CACHE_DIR)
    home_dir = osp.expanduser(home_dir)
    return home_dir


def set_home_dir(path: str):
    r"""Set the cache directory used for storing all PyG-related data.

    Args:
        path (str): The path to a local folder.
    """
    global _home_dir
    _home_dir = path
