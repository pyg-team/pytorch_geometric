import errno
import os
import os.path as osp
<<<<<<< HEAD
from pathlib import Path
import errno
=======
>>>>>>> 4557254c849eda62ce1860a56370eb4a54aa76dd


def makedirs(path: str):
    r"""Recursive directory creation function."""
    try:
        Path.expanduser(Path(path).resolve()).mkdir(parents=True, exist_ok=True)
        # os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and Path(path).is_dir():
            raise e
