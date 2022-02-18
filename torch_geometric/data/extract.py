import bz2
import gzip
import os.path as osp
import sys
import tarfile
import zipfile


def maybe_log(path, log=True):
    if log:
        print(f'Extracting {path}', file=sys.stderr)


def extract_tar(path: str, folder: str, mode: str = 'r:gz', log: bool = True):
    r"""Extracts a tar archive to a specific folder.

    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        mode (string, optional): The compression mode. (default: :obj:`"r:gz"`)
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    maybe_log(path, log)
    with tarfile.open(path, mode) as f:
        f.extractall(folder)


def extract_zip(path: str, folder: str, log: bool = True):
    r"""Extracts a zip archive to a specific folder.

    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    maybe_log(path, log)
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)


def extract_bz2(path: str, folder: str, log: bool = True):
    r"""Extracts a bz2 archive to a specific folder.

    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    maybe_log(path, log)
    path = osp.abspath(path)
    with bz2.open(path, 'r') as r:
        with open(osp.join(folder, '.'.join(path.split('.')[:-1])), 'wb') as w:
            w.write(r.read())


def extract_gz(path: str, folder: str, log: bool = True):
    r"""Extracts a gz archive to a specific folder.

    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    maybe_log(path, log)
    path = osp.abspath(path)
    with gzip.open(path, 'r') as r:
        with open(osp.join(folder, '.'.join(path.split('.')[:-1])), 'wb') as w:
            w.write(r.read())
