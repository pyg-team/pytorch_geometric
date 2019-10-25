import sys
import random
import os.path as osp
import shutil

from torch_geometric.datasets import TUDataset
import pytest


def test_noniso_mutag():
    root = osp.join('/', 'tmp', str(random.randrange(sys.maxsize)))
    dataset = TUDataset(root, 'MUTAG', cleaned=True)

    assert len(dataset) == 135
    assert dataset.num_features == 7
    assert dataset.num_classes == 2
    assert dataset.__repr__() == 'MUTAG(135)'

    assert len(dataset[0]) == 4
    assert dataset[0].edge_attr.size(1) == 4

    dataset = TUDataset(root, 'MUTAG', use_node_attr=True, cleaned=True)
    assert dataset.num_features == 7

    shutil.rmtree(root)
