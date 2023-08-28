import os
import os.path as osp
import random
import sys

import torch

from torch_geometric.data import Data
from torch_geometric.io import read_off, write_off


def test_read_off():
    root_dir = osp.join(osp.dirname(osp.realpath(__file__)))

    data = read_off(osp.join(root_dir, 'example1.off'))
    assert len(data) == 2
    assert data.pos.tolist() == [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]
    assert data.face.tolist() == [[0, 1], [1, 2], [2, 3]]

    data = read_off(osp.join(root_dir, 'example2.off'))
    assert len(data) == 2
    assert data.pos.tolist() == [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]
    assert data.face.tolist() == [[0, 0], [1, 2], [2, 3]]


def test_write_off():
    pos = torch.tensor([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
    face = torch.tensor([[0, 1], [1, 2], [2, 3]])

    name = str(random.randrange(sys.maxsize))
    path = osp.join('/', 'tmp', f'{name}.off')
    write_off(Data(pos=pos, face=face), path)
    data = read_off(path)
    os.unlink(path)

    assert data.pos.tolist() == pos.tolist()
    assert data.face.tolist() == face.tolist()
