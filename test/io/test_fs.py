import zipfile
from os import path as osp

import fsspec

from torch_geometric.io import fs


def test_get_fs():
    assert 'file' in fs.get_fs('/tmp/test').protocol
    assert 'memory' in fs.get_fs('memory:///tmp/test').protocol


def test_normpath():
    assert fs.normpath('////home') == '/home'
    assert fs.normpath('memory:////home') == 'memory:////home'


def test_exists(tmp_path):
    path = osp.join(tmp_path, 'file.txt')
    assert not fs.exists(path)
    with fsspec.open(path, 'w') as f:
        f.write('here')
    assert fs.exists(path)


def test_makedirs(tmp_path):
    path = osp.join(tmp_path, '1', '2')
    assert not fs.isdir(path)
    fs.makedirs(path)
    assert fs.isdir(path)


def test_ls(tmp_path):
    tmp_path = tmp_path.resolve().as_posix()

    for i in range(2):
        with fsspec.open(osp.join(tmp_path, str(i)), 'w') as f:
            f.write('here')
    res = fs.ls(tmp_path)
    assert len(res) == 2
    expected_protocol = fs.get_fs(tmp_path).protocol
    assert all(fs.get_fs(path).protocol == expected_protocol for path in res)


def test_cp(tmp_path):
    src = osp.join(tmp_path, 'src')
    for i in range(2):
        with fsspec.open(osp.join(src, str(i)), 'w') as f:
            f.write('here')
    assert fs.exists(src)

    dst = osp.join(tmp_path, 'dst')
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


def test_cp_extract(tmp_path):
    def make_zip(path: str):
        with fsspec.open(path, mode='wb') as f:
            with zipfile.ZipFile(f, mode='w') as z:
                z.writestr('1', b'data')
                z.writestr('2', b'data')

    src = osp.join(tmp_path, 'src', 'test.zip')
    make_zip(src)
    assert len(fsspec.open_files(f'zip://*::{src}')) == 2

    dst = osp.join(tmp_path, 'dst')
    assert not fs.exists(dst)

    # Can copy:
    fs.makedirs(dst)
    fs.cp(src, dst)
    assert fs.exists(osp.join(dst, 'test.zip'))

    # Can copy and extract:
    fs.cp(src, dst, extract=True)
    for i in range(2):
        fs.exists(osp.join(dst, str(i)))
