import io
import os.path as osp
import sys
from typing import Any, Dict, List, Optional, Union

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
    if get_fs(path).protocol == 'file':
        return osp.normpath(path)
    return path


def exists(path: str) -> bool:
    return get_fs(path).exists(path)


def makedirs(path: str, exist_ok: bool = True):
    return get_fs(path).makedirs(path, exist_ok)


def isdir(path: str) -> bool:
    return get_fs(path).isdir(path)


def ls(
    path: str,
    detail: bool = False,
) -> Union[List[str], List[Dict[str, Any]]]:
    return get_fs(path).ls(path, detail=detail)


def cp(
    path1: str,
    path2: str,
    extract: bool = False,
    kwargs1: Optional[dict] = None,
    kwargs2: Optional[dict] = None,
    log: bool = True,
):
    kwargs1 = kwargs1 or {}
    kwargs2 = kwargs2 or {}

    # Cache result if the protocol is not local:
    cache_dir: Optional[str] = None
    if get_fs(path1).protocol not in {'file', 'memory'}:
        if log and 'pytest' not in sys.modules:
            print(f'Downloading {path1}', file=sys.stderr)

        cache_dir = osp.join(torch_geometric.get_home_dir(), 'simplecache')
        kwargs1.setdefault('simplecache', dict(cache_storage=cache_dir))
        path1 = f'simplecache::{path1}'

    # Extract = Unarchive + Decompress if applicable.
    if extract and path1.endswith('.tar.gz'):
        kwargs1.setdefault('tar', dict(compression='gzip'))
        path1 = f'tar://**::{path1}'
    elif extract and path1.endswith('.zip'):
        path1 = f'zip://**::{path1}'
    elif extract:
        raise NotImplementedError(f"Automatic extraction of '{path1}' not "
                                  f"yet supported")

    def _copy_file_handle(f_from: Any, path: str, **kwargs):
        with fsspec.open(path, 'wb', **kwargs) as f_to:
            while True:
                chunk = f_from.read(10 * 1024 * 1024)
                if not chunk:
                    break
                f_to.write(chunk)

    if extract:
        open_files = fsspec.open_files(path1, **kwargs1)
        with open_files as of:
            for f_from, open_file in zip(of, open_files):
                to_path = osp.join(path2, open_file.path)
                _copy_file_handle(f_from, to_path, **kwargs2)
    else:
        if isdir(path2):
            path2 = osp.join(path2, osp.basename(path1))

        with fsspec.open(path1, **kwargs1) as f_from:
            _copy_file_handle(f_from, path2, **kwargs2)

    if cache_dir is not None:
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
    return fs.glob(path)


def torch_save(data: Any, path: str):
    buffer = io.BytesIO()
    torch.save(data, buffer)
    with fsspec.open(path, 'wb') as f:
        f.write(buffer.getvalue())


def torch_load(path: str, map_location: Any = None) -> Any:
    with fsspec.open(path, 'rb') as f:
        return torch.load(f, map_location)
