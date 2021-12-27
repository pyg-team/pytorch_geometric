import os
import os.path as osp
from pathlib import Path
import errno


def makedirs(path: str):
    r"""Recursive directory creation function."""
    try:
        Path.expanduser(Path(path).resolve()).mkdir(parents=True, exist_ok=True)
        # os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and Path(path).is_dir():
            raise e
