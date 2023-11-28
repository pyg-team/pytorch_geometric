from collections.abc import Callable
import pytest
import fsspec
import functools
import zipfile
from os import path as osp

from torch_geometric.io import fs


@pytest.fixture()
def _tmp_path() -> str:
    support_memory_fs = False

    # TODO Support memory filesystem on Windows.
    # TODO Support memory filesystem for all datasets.
    if not support_memory_fs:
        root = osp.join('/', 'tmp', 'pyg_test_fs')
    else:
        root = 'memory://pyg_test_fs'

    yield root

    if fs.exists(root):
        fs.rm(root)


def _make_tmp_zip(path: str, data: dict):
    data = data or {}
    with fsspec.open(path, mode='wb') as f:
        with zipfile.ZipFile(f, mode="w") as z:
            for k, v in data.items():
                z.writestr(k, v)
    return path


def _get_protocol(path: str) -> str:
    return fs.get_fs(path).protocol


def test_fs_get_fs():
    assert 'file' in _get_protocol('/tmp/test')
    assert 'memory' in _get_protocol('memory:///tmp/test')


def test_fs_normpath_for_local_only():
    assert fs.normpath('////home') == '/home'
    assert fs.normpath('memory:////home') == 'memory:////home'


def test_fs_exists(_tmp_path):
    assert not fs.exists(_tmp_path)
    with fsspec.open(_tmp_path, 'wt') as f:
        f.write('here')
    assert fs.exists(_tmp_path)


def test_fs_makedirs(_tmp_path):
    path = osp.join(_tmp_path, '1', '2')
    assert not fs.isdir(path)
    fs.makedirs(path)
    assert fs.isdir(path)


def test_fs_ls(_tmp_path):
    assert not fs.exists(_tmp_path)
    num_files = 2
    for n in range(num_files):
        path = osp.join(_tmp_path, str(n))
        with fsspec.open(path, 'wt') as f:
            f.write('here')
    res = fs.ls(_tmp_path)
    assert len(res) == num_files
    assert all(_get_protocol(p) == _get_protocol(_tmp_path) for p in res)


def test_fs_copy(_tmp_path):
    src, dst = [osp.join(_tmp_path, f) for f in ['src', 'dst']]
    num_files = 2
    for n in range(num_files):
        path = osp.join(src, str(n))
        with fsspec.open(path, 'wt') as f:
            f.write('here')
    assert not fs.exists(dst)
    fs.makedirs(dst)
    fs.cp(osp.join(src, '1'), dst)
    assert len(fs.ls(dst)) == 1
    fs.cp(osp.join(src, '*'), dst)
    assert len(fs.ls(dst)) == 2


def test_fs_copy_extract(_tmp_path):
    src, dst = [osp.join(_tmp_path, f) for f in ['src', 'dst']]
    src = osp.join(src, 'test.zip')

    # Create some zip with test files.
    num_files = 2
    zip_data = {str(k): b'data' for k in range(num_files)}
    src = _make_tmp_zip(src, zip_data)
    assert len(fsspec.open_files(f'zip://*::{src}')) == num_files

    # Can copy the zip file.
    assert not fs.exists(dst)
    fs.makedirs(dst)
    fs.cp(src, dst)
    assert fs.exists(osp.join(dst, 'test.zip'))

    # Can copy and extract.
    fs.cp(src, dst, extract=True)
    for n in range(num_files):
        fs.exists(osp.join(dst, str(n)))
