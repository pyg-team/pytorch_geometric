import pytest
import torch

from torch_geometric.loader import DataListLoader, DataLoader, DenseDataLoader
from torch_geometric.testing import onlyOnline
from torch_geometric.transforms import ToDense


@onlyOnline
def test_enzymes(get_dataset):
    dataset = get_dataset(name='ENZYMES')
    assert len(dataset) == 600
    assert dataset.num_features == 3
    assert dataset.num_classes == 6
    assert str(dataset) == 'ENZYMES(600)'

    assert len(dataset[0]) == 3
    assert len(dataset.shuffle()) == 600
    assert len(dataset.shuffle(return_perm=True)) == 2
    assert len(dataset[:100]) == 100
    assert len(dataset[0.1:0.2]) == 60
    assert len(dataset[torch.arange(100, dtype=torch.long)]) == 100
    mask = torch.zeros(600, dtype=torch.bool)
    mask[:100] = 1
    assert len(dataset[mask]) == 100

    loader = DataLoader(dataset, batch_size=len(dataset))
    for batch in loader:
        assert batch.num_graphs == len(batch) == 600

        avg_num_nodes = batch.num_nodes / batch.num_graphs
        assert pytest.approx(avg_num_nodes, abs=1e-2) == 32.63

        avg_num_edges = batch.num_edges / (2 * batch.num_graphs)
        assert pytest.approx(avg_num_edges, abs=1e-2) == 62.14

        assert list(batch.x.size()) == [batch.num_nodes, 3]
        assert list(batch.y.size()) == [batch.num_graphs]
        assert batch.y.max() + 1 == 6
        assert list(batch.batch.size()) == [batch.num_nodes]
        assert batch.ptr.numel() == batch.num_graphs + 1

        assert batch.has_isolated_nodes()
        assert not batch.has_self_loops()
        assert batch.is_undirected()

    loader = DataListLoader(dataset, batch_size=len(dataset))
    for data_list in loader:
        assert len(data_list) == 600

    dataset.transform = ToDense(num_nodes=126)
    loader = DenseDataLoader(dataset, batch_size=len(dataset))
    for data in loader:
        assert list(data.x.size()) == [600, 126, 3]
        assert list(data.adj.size()) == [600, 126, 126]
        assert list(data.mask.size()) == [600, 126]
        assert list(data.y.size()) == [600, 1]


@onlyOnline
def test_enzymes_with_node_attr(get_dataset):
    dataset = get_dataset(name='ENZYMES', use_node_attr=True)
    assert dataset.num_node_features == 21
    assert dataset.num_features == 21
    assert dataset.num_edge_features == 0


@onlyOnline
def test_cleaned_enzymes(get_dataset):
    dataset = get_dataset(name='ENZYMES', cleaned=True)
    assert len(dataset) == 595
