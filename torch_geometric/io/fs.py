import io
import os.path as osp
import sys
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import fsspec
import torch

import torch_geometric

DEFAULT_CACHE_PATH = '/tmp/pyg_simplecache'


def get_fs(path: str) -> fsspec.AbstractFileSystem:
    r"""Get filesystem backend given a path URI to the resource.

    Here are some common example paths and dispatch result:

    * :obj:`"/home/file"` ->
      :class:`fsspec.implementations.local.LocalFileSystem`
    * :obj:`"memory://home/file"` ->
      :class:`fsspec.implementations.memory.MemoryFileSystem`
    * :obj:`"https://home/file"` ->
      :class:`fsspec.implementations.http.HTTPFileSystem`
    * :obj:`"gs://home/file"` -> :class:`gcsfs.GCSFileSystem`
    * :obj:`"s3://home/file"` -> :class:`s3fs.S3FileSystem`

    A full list of supported backend implementations of :class:`fsspec` can be
    found `here <https://github.com/fsspec/filesystem_spec/blob/master/fsspec/
    registry.py#L62>`_.

    The backend dispatch logic can be updated with custom backends following
    `this tutorial <https://filesystem-spec.readthedocs.io/en/latest/
    developer.html#implementing-a-backend>`_.

    Args:
        path (str): The URI to the filesystem location, *e.g.*,
            :obj:`"gs://home/me/file"`, :obj:`"s3://..."`.
    """
    return fsspec.core.url_to_fs(path)[0]


def normpath(path: str) -> str:
    if isdisk(path):
        return osp.normpath(path)
    return path


def exists(path: str) -> bool:
    return get_fs(path).exists(path)


def makedirs(path: str, exist_ok: bool = True):
    return get_fs(path).makedirs(path, exist_ok)


def isdir(path: str) -> bool:
    return get_fs(path).isdir(path)


def isfile(path: str) -> bool:
    return get_fs(path).isfile(path)


def isdisk(path: str) -> bool:
    return 'file' in get_fs(path).protocol


def islocal(path: str) -> bool:
    return isdisk(path) or 'memory' in get_fs(path).protocol


def ls(
    path: str,
    detail: bool = False,
) -> Union[List[str], List[Dict[str, Any]]]:
    fs = get_fs(path)
    paths = fs.ls(path, detail=detail)

    if not isdisk(path):
        if detail:
            for path in paths:
                path['name'] = fs.unstrip_protocol(path['name'])
        else:
            paths = [fs.unstrip_protocol(path) for path in paths]

    return paths


def cp(
    path1: str,
    path2: str,
    extract: bool = False,
    log: bool = True,
    use_cache: bool = True,
    clear_cache: bool = True,
):
    kwargs = {}

    # Cache result if the protocol is not local:
    cache_dir: Optional[str] = None
    if not islocal(path1):
        if log and 'pytest' not in sys.modules:
            print(f'Downloading {path1}', file=sys.stderr)

        if use_cache:  # Cache seems to confuse the gcs filesystem.
            home_dir = torch_geometric.get_home_dir()
            cache_dir = osp.join(home_dir, 'simplecache', uuid4().hex)
            kwargs.setdefault('simplecache', dict(cache_storage=cache_dir))
            path1 = f'simplecache::{path1}'

    # Handle automatic extraction:
    if extract and path1.endswith('.tar.gz'):
        kwargs.setdefault('tar', dict(compression='gzip'))
        path1 = f'tar://**::{path1}'
    elif extract and path1.endswith('.zip'):
        path1 = f'zip://**::{path1}'
    elif extract:
        raise NotImplementedError(
            f"Automatic extraction of '{path1}' not yet supported")

    # If the source path points to a directory, we need to make sure to
    # recursively copy all files within this directory. Additionally, if the
    # destination folder does not yet exist, we inherit the basename from the
    # source folder.
    if isdir(path1):
        if exists(path2):
            path2 = osp.join(path2, osp.basename(path1))
        path1 = osp.join(path1, '**')

    # Perform the copy:
    for open_file in fsspec.open_files(path1, **kwargs):
        with open_file as f_from:
            if isfile(path1):
                if isdir(path2):
                    to_path = osp.join(path2, osp.basename(path1))
                else:
                    to_path = path2
            else:
                # Open file has protocol stripped.
                common_path = osp.commonprefix(
                    [fsspec.core.strip_protocol(path1), open_file.path])
                to_path = osp.join(path2, open_file.path[len(common_path):])
            with fsspec.open(to_path, 'wb') as f_to:
                while True:
                    chunk = f_from.read(10 * 1024 * 1024)
                    if not chunk:
                        break
                    f_to.write(chunk)

    if use_cache and clear_cache and cache_dir is not None:
        rm(cache_dir)


def rm(path: str, recursive: bool = True):
    get_fs(path).rm(path, recursive)


def mv(path1: str, path2: str, recursive: bool = True):
    fs1 = get_fs(path1)
    fs2 = get_fs(path2)
    assert fs1.protocol == fs2.protocol
    fs1.mv(path1, path2, recursive)


def glob(path: str):
    fs = get_fs(path)
    paths = fs.glob(path)

    if not isdisk(path):
        paths = [fs.unstrip_protocol(path) for path in paths]

    return paths


def torch_save(data: Any, path: str):
    buffer = io.BytesIO()
    torch.save(data, buffer)
    with fsspec.open(path, 'wb') as f:
        f.write(buffer.getvalue())


def torch_load(path: str, map_location: Any = None) -> Any:
    with fsspec.open(path, 'rb') as f:
        return torch.load(f, map_location)
