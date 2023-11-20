import io
import os.path as osp
import sys
from typing import Any, Optional

import fsspec
import torch
from fsspec.core import url_to_fs

DEFAULT_CACHE_PATH = '/tmp/pyg_simplecache'


def get_fs(path: str) -> fsspec.AbstractFileSystem:
    return url_to_fs(path)[0]


def fs_normpath(path: str, *args) -> str:
    if get_fs(path).protocol == 'file':
        return osp.normpath(path, *args)
    return path


def fs_exists(path: str) -> bool:
    return get_fs(path).exists(path)


def fs_ls(path: str, detail: bool = False) -> bool:
    fs = get_fs(path)
    results = fs.ls(path, detail=detail)
    # TODO: Strip common ancestor.
    return [osp.basename(x) for x in results]


def fs_mkdirs(path: str, **kwargs):
    return get_fs(path).mkdirs(path, **kwargs)


def fs_isdir(path: str) -> bool:
    return get_fs(path).isdir(path)


def _copy_file_handle(f_from: Any, path: str, blocksize: int, **kwargs):
    with fsspec.open(path, 'wb', **kwargs) as f_to:
        data = True
        while data:
            data = f_from.read(blocksize)
            f_to.write(data)


def fs_cp(path1: str, path2: str, kwargs1: Optional[dict] = None,
          kwargs2: Optional[dict] = None, extract: bool = False,
          cache_path: Optional[str] = DEFAULT_CACHE_PATH,
          blocksize: int = 2 * 22):
    # Initialize kwargs for source/destination.
    kwargs1 = kwargs1 or {}
    kwargs2 = kwargs2 or {}

    # Cache result if the protocol is not local and we have a cache folder.
    if get_fs(path1).protocol != 'file' and cache_path:
        kwargs1 = {
            **kwargs1,
            'simplecache': {
                'cache_storage': cache_path
            },
        }
        path1 = f'simplecache::{path1}'

    # Extract = Unarchive + Decompress if applicable.
    if extract and path1.endswith('.tar.gz'):
        kwargs1 = {**kwargs1, 'tar': {'compression': 'gzip'}}
        path1 = f'tar://**::{path1}'
    elif extract and path1.endswith('.zip'):
        path1 = f'zip://**::{path1}'
    else:
        name = osp.basename(path1)
        if extract and name.endswith('.gz') or name.endswith('.bz2'):
            name = osp.splitext(name)[0]
        path2 = osp.join(path2, name)

    if '*' in path1:
        open_files = fsspec.open_files(path1, **kwargs1)
        with open_files as of:
            for f_from, open_file in zip(of, open_files):
                to_path = osp.join(path2, open_file.path)
                _copy_file_handle(f_from, to_path, blocksize, **kwargs2)
    else:
        with fsspec.open(path1, compression='infer', **kwargs1) as f_from:
            _copy_file_handle(f_from, path2, blocksize, **kwargs2)


def fs_rm(path: str, **kwargs):
    get_fs(path).rm(path, **kwargs)


def fs_mv(path1: str, path2: str, **kwargs):
    fs1 = get_fs(path1)
    fs2 = get_fs(path2)
    assert fs1.protocol == fs2.protocol
    fs1.mv(path1, path2, **kwargs)


def fs_glob(path: str, **kwargs):
    fs = get_fs(path)
    return [fs.unstrip_protocol(x) for x in fs.glob(path, **kwargs)]


def fs_torch_save(data: Any, path: str):
    buffer = io.BytesIO()
    torch.save(data, buffer)
    with fsspec.open(path, 'wb') as f:
        f.write(buffer.getvalue())


def fs_torch_load(path: str, map_location: Any = None) -> Any:
    with fsspec.open(path, 'rb') as f:
        return torch.load(f, map_location)
