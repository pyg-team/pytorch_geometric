import io
import os.path as osp
import pickle
import re
import sys
import warnings
from typing import Any, Dict, List, Literal, Optional, Union, overload
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


def makedirs(path: str, exist_ok: bool = True) -> None:
    return get_fs(path).makedirs(path, exist_ok)


def isdir(path: str) -> bool:
    return get_fs(path).isdir(path)


def isfile(path: str) -> bool:
    return get_fs(path).isfile(path)


def isdisk(path: str) -> bool:
    return 'file' in get_fs(path).protocol


def islocal(path: str) -> bool:
    return isdisk(path) or 'memory' in get_fs(path).protocol


@overload
def ls(path: str, detail: Literal[False] = False) -> List[str]:
    pass


@overload
def ls(path: str, detail: Literal[True]) -> List[Dict[str, Any]]:
    pass


def ls(
    path: str,
    detail: bool = False,
) -> Union[List[str], List[Dict[str, Any]]]:
    fs = get_fs(path)
    outputs = fs.ls(path, detail=detail)

    if not isdisk(path):
        if detail:
            for output in outputs:
                output['name'] = fs.unstrip_protocol(output['name'])
        else:
            outputs = [fs.unstrip_protocol(output) for output in outputs]

    return outputs


def cp(
    path1: str,
    path2: str,
    extract: bool = False,
    log: bool = True,
    use_cache: bool = True,
    clear_cache: bool = True,
) -> None:
    kwargs: Dict[str, Any] = {}

    is_path1_dir = isdir(path1)
    is_path2_dir = isdir(path2)

    # Cache result if the protocol is not local:
    cache_dir: Optional[str] = None
    if not islocal(path1):
        if log and 'pytest' not in sys.modules:
            print(f'Downloading {path1}', file=sys.stderr)

        if extract and use_cache:  # Cache seems to confuse the gcs filesystem.
            home_dir = torch_geometric.get_home_dir()
            cache_dir = osp.join(home_dir, 'simplecache', uuid4().hex)
            kwargs.setdefault('simplecache', dict(cache_storage=cache_dir))
            path1 = f'simplecache::{path1}'

    # Handle automatic extraction:
    multiple_files = False
    if extract and path1.endswith('.tar.gz'):
        kwargs.setdefault('tar', dict(compression='gzip'))
        path1 = f'tar://**::{path1}'
        multiple_files = True
    elif extract and path1.endswith('.zip'):
        path1 = f'zip://**::{path1}'
        multiple_files = True
    elif extract and path1.endswith('.gz'):
        kwargs.setdefault('compression', 'infer')
    elif extract:
        raise NotImplementedError(
            f"Automatic extraction of '{path1}' not yet supported")

    # If the source path points to a directory, we need to make sure to
    # recursively copy all files within this directory. Additionally, if the
    # destination folder does not yet exist, we inherit the basename from the
    # source folder.
    if is_path1_dir:
        if exists(path2):
            path2 = osp.join(path2, osp.basename(path1))
        path1 = osp.join(path1, '**')
        multiple_files = True

    # Perform the copy:
    for open_file in fsspec.open_files(path1, **kwargs):
        with open_file as f_from:
            if not multiple_files:
                if is_path2_dir:
                    basename = osp.basename(path1)
                    if extract and path1.endswith('.gz'):
                        basename = '.'.join(basename.split('.')[:-1])
                    to_path = osp.join(path2, basename)
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
        try:
            rm(cache_dir)
        except Exception:  # FIXME
            # Windows test yield "PermissionError: The process cannot access
            # the file because it is being used by another process".
            # Users may also observe "OSError: Directory not empty".
            # This is a quick workaround until we figure out the deeper issue.
            pass


def rm(path: str, recursive: bool = True) -> None:
    get_fs(path).rm(path, recursive)


def mv(path1: str, path2: str) -> None:
    fs1 = get_fs(path1)
    fs2 = get_fs(path2)
    assert fs1.protocol == fs2.protocol
    fs1.mv(path1, path2)


def glob(path: str) -> List[str]:
    fs = get_fs(path)
    paths = fs.glob(path)

    if not isdisk(path):
        paths = [fs.unstrip_protocol(path) for path in paths]

    return paths


def torch_save(data: Any, path: str) -> None:
    buffer = io.BytesIO()
    torch.save(data, buffer)
    with fsspec.open(path, 'wb') as f:
        f.write(buffer.getvalue())


def torch_load(path: str, map_location: Any = None) -> Any:
    if torch_geometric.typing.WITH_PT24:
        try:
            with fsspec.open(path, 'rb') as f:
                return torch.load(f, map_location, weights_only=True)
        except pickle.UnpicklingError as e:
            error_msg = str(e)
            if "add_safe_globals" in error_msg:
                warn_msg = ("Weights only load failed. Please file an issue "
                            "to make `torch.load(weights_only=True)` "
                            "compatible in your case.")
                match = re.search(r'add_safe_globals\(.*?\)', error_msg)
                if match is not None:
                    warnings.warn(f"{warn_msg} Please use "
                                  f"`torch.serialization.{match.group()}` to "
                                  f"allowlist this global.")
                else:
                    warnings.warn(warn_msg)

                with fsspec.open(path, 'rb') as f:
                    return torch.load(f, map_location, weights_only=False)
            else:
                raise e

    with fsspec.open(path, 'rb') as f:
        return torch.load(f, map_location)
