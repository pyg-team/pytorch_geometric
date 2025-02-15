import zipfile
from os import path as osp

import fsspec
import pytest
import torch

import torch_geometric.typing
from torch_geometric.data import extract_zip
from torch_geometric.io import fs
from torch_geometric.testing import noWindows

if torch_geometric.typing.WITH_WINDOWS:  # FIXME
    params = ['file']
else:
    params = ['file', 'memory']


@pytest.fixture(params=params)
def tmp_fs_path(request, tmp_path) -> str:
    if request.param == 'file':
        return tmp_path.resolve().as_posix()
    elif request.param == 'memory':
        return f'memory://{tmp_path}'
    raise NotImplementedError


def test_get_fs():
    assert 'file' in fs.get_fs('/tmp/test').protocol
    assert 'memory' in fs.get_fs('memory:///tmp/test').protocol


@noWindows
def test_normpath():
    assert fs.normpath('////home') == '/home'
    assert fs.normpath('memory:////home') == 'memory:////home'


def test_exists(tmp_fs_path):
    path = osp.join(tmp_fs_path, 'file.txt')
    assert not fs.exists(path)
    with fsspec.open(path, 'w') as f:
        f.write('here')
    assert fs.exists(path)


def test_makedirs(tmp_fs_path):
    path = osp.join(tmp_fs_path, '1', '2')
    assert not fs.isdir(path)
    fs.makedirs(path)
    assert fs.isdir(path)


@pytest.mark.parametrize('detail', [False, True])
def test_ls(tmp_fs_path, detail):
    for i in range(2):
        with fsspec.open(osp.join(tmp_fs_path, str(i)), 'w') as f:
            f.write('here')
    res = fs.ls(tmp_fs_path, detail)
    assert len(res) == 2
    expected_protocol = fs.get_fs(tmp_fs_path).protocol
    for output in res:
        if detail:
            output = output['name']
        assert fs.get_fs(output).protocol == expected_protocol


def test_cp(tmp_fs_path):
    src = osp.join(tmp_fs_path, 'src')
    for i in range(2):
        with fsspec.open(osp.join(src, str(i)), 'w') as f:
            f.write('here')
    assert fs.exists(src)

    dst = osp.join(tmp_fs_path, 'dst')
    assert not fs.exists(dst)

    # Can copy a file to new name:
    fs.cp(osp.join(src, '1'), dst)
    assert fs.isfile(dst)
    fs.rm(dst)

    # Can copy a single file to directory:
    fs.makedirs(dst)
    fs.cp(osp.join(src, '1'), dst)
    assert len(fs.ls(dst)) == 1

    # Can copy multiple files to directory:
    fs.cp(src, dst)
    assert len(fs.ls(dst)) == 2
    for i in range(2):
        fs.exists(osp.join(dst, str(i)))


def test_extract(tmp_fs_path):
    def make_zip(path: str):
        with fsspec.open(path, mode='wb') as f:
            with zipfile.ZipFile(f, mode='w') as z:
                z.writestr('1', b'data')
                z.writestr('2', b'data')

    src = osp.join(tmp_fs_path, 'src', 'test.zip')
    make_zip(src)
    assert len(fsspec.open_files(f'zip://*::{src}')) == 2

    dst = osp.join(tmp_fs_path, 'dst')
    assert not fs.exists(dst)

    # Can copy and extract afterwards:
    if fs.isdisk(tmp_fs_path):
        fs.cp(src, osp.join(dst, 'test.zip'))
        assert fs.exists(osp.join(dst, 'test.zip'))
        extract_zip(osp.join(dst, 'test.zip'), dst)
        assert len(fs.ls(dst)) == 3
        for i in range(2):
            fs.exists(osp.join(dst, str(i)))
        fs.rm(dst)

    # Can copy and extract:
    fs.cp(src, dst, extract=True)
    assert len(fs.ls(dst)) == 2
    for i in range(2):
        fs.exists(osp.join(dst, str(i)))


def test_torch_save_load(tmp_fs_path):
    x = torch.randn(5, 5)
    path = osp.join(tmp_fs_path, 'x.pt')

    fs.torch_save(x, path)
    out = fs.torch_load(path)
    assert torch.equal(x, out)
