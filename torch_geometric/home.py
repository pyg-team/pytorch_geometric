import os
from pathlib import Path

ENV_PYG_HOME = 'PYG_HOME'
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "pyg"

_home_dir = None


def get_home_dir() -> str:
    r"""Get the cache directory used for storing all :pyg:`PyG`-related data.

    If :meth:`set_home_dir` is not called, the path is given by the environment
    variable :obj:`$PYG_HOME` which defaults to :obj:`"~/.cache/pyg"`.
    """
    return _home_dir if _home_dir is not None else os.path.expanduser(os.path.expandvars("$PYG_HOME") or DEFAULT_CACHE_DIR)


def set_home_dir(path: str):
    r"""Set the cache directory used for storing all :pyg:`PyG`-related data.

    Args:
        path (str): The path to a local folder.
    """
    global _home_dir
    _home_dir = path
