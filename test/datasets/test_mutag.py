import sys
import random
import os.path as osp
import shutil

from torch_geometric.datasets import TUDataset


def test_mutag():
    root = osp.join('/', 'tmp', str(random.randrange(sys.maxsize)))
    dataset = TUDataset(root, 'MUTAG')

    assert len(dataset) == 188
    assert dataset.num_features == 7
    assert dataset.num_classes == 2
    assert dataset.__repr__() == 'MUTAG(188)'

    assert len(dataset[0]) == 3
    assert dataset[0].edge_attr.size(1) == 4

    dataset = TUDataset(root, 'MUTAG', use_node_attributes=True)
    assert dataset.num_features == 7

    shutil.rmtree(root)
