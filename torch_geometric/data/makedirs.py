# import os.path as
from pathlib import Path
import errno


def makedirs(path):
    try:
        Path.mkdir(Path.expanduser(Path(str(path)).resolve()))
        # os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and Path(path).is_dir():
            raise e
