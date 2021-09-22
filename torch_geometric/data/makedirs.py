# import os.path as
from pathlib import Path
import errno


def makedirs(path):
    try:
        Path.mkdir(Path.expanduser(Path(path).resolve()))
        # os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and Path.isdir(path):
            raise e
