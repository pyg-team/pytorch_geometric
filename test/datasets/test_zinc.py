import sys
import random
import os.path as osp
import shutil

from torch_geometric.datasets import ZINC


def test_mutag():
    root = osp.join('/', 'tmp', str(random.randrange(sys.maxsize)))
    dataset = ZINC(root)

    print(len(dataset))

    shutil.rmtree(root)
