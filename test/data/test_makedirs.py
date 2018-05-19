import sys
import random
import os.path as osp
import shutil

from torch_geometric.data import makedirs


def test_makedirs():
    path = osp.join('/', 'tmp', str(random.randrange(sys.maxsize)), 'test')
    assert not osp.exists(path)
    makedirs(path)
    makedirs(path)
    assert osp.exists(path)
    shutil.rmtree(path)
