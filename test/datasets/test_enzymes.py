import os.path as osp
import random
import shutil
import sys

import pytest
import torch

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataListLoader, DataLoader, DenseDataLoader
from torch_geometric.transforms import ToDense


def test_enzymes():
    root = osp.join('/', 'tmp', str(random.randrange(sys.maxsize)))
    dataset = TUDataset(root, 'ENZYMES')

    assert len(dataset) == 600
    assert dataset.num_features == 3
    assert dataset.num_classes == 6
    assert dataset.__repr__() == 'ENZYMES(600)'

    assert len(dataset[0]) == 3
    assert len(dataset.shuffle()) == 600
    assert len(dataset.shuffle(return_perm=True)) == 2
    assert len(dataset[:100]) == 100
    assert len(dataset[torch.arange(100, dtype=torch.long)]) == 100
    mask = torch.zeros(600, dtype=torch.bool)
    mask[:100] = 1
    assert len(dataset[mask]) == 100

    loader = DataLoader(dataset, batch_size=len(dataset))
    for data in loader:
        assert data.num_graphs == 600

        avg_num_nodes = data.num_nodes / data.num_graphs
        assert pytest.approx(avg_num_nodes, abs=1e-2) == 32.63

        avg_num_edges = data.num_edges / (2 * data.num_graphs)
        assert pytest.approx(avg_num_edges, abs=1e-2) == 62.14

        assert len(data) == 5
        assert list(data.x.size()) == [data.num_nodes, 3]
        assert list(data.y.size()) == [data.num_graphs]
        assert data.y.max() + 1 == 6
        assert list(data.batch.size()) == [data.num_nodes]
        assert data.ptr.numel() == data.num_graphs + 1

        assert data.has_isolated_nodes()
        assert not data.has_self_loops()
        assert data.is_undirected()

    loader = DataListLoader(dataset, batch_size=len(dataset))
    for data_list in loader:
        assert len(data_list) == 600

    dataset.transform = ToDense(num_nodes=126)
    loader = DenseDataLoader(dataset, batch_size=len(dataset))
    for data in loader:
        assert len(data) == 4
        assert list(data.x.size()) == [600, 126, 3]
        assert list(data.adj.size()) == [600, 126, 126]
        assert list(data.mask.size()) == [600, 126]
        assert list(data.y.size()) == [600, 1]

    dataset = TUDataset(root, 'ENZYMES', use_node_attr=True)
    assert dataset.num_node_features == 21
    assert dataset.num_features == 21
    assert dataset.num_edge_features == 0

    shutil.rmtree(root)


def test_cleaned_enzymes():
    root = osp.join('/', 'tmp', str(random.randrange(sys.maxsize)))
    dataset = TUDataset(root, 'ENZYMES', cleaned=True)

    assert len(dataset) == 595

    shutil.rmtree(root)
