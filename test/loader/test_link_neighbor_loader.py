import pytest
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import LinkNeighborLoader


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.long)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.long)
    return torch.stack([row, col], dim=0)


def unique_edge_pairs(edge_index):
    return set(map(tuple, edge_index.t().tolist()))


@pytest.mark.parametrize('directed', [True, False])
@pytest.mark.parametrize('labeled', [True, False])
def test_homogeneous_link_neighbor_loader(directed, labeled):
    torch.manual_seed(12345)

    pos_edge_index = get_edge_index(100, 50, 500)
    neg_edge_index = get_edge_index(100, 50, 500)
    neg_edge_index[1, :] += 50

    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([torch.ones(500), torch.zeros(500)], dim=0)
    if not labeled:
        edge_label = None

    data = Data()

    data.edge_index = pos_edge_index
    data.x = torch.arange(100)
    data.edge_attr = torch.arange(500)

    loader = LinkNeighborLoader(data, num_neighbors=[-1] * 2, batch_size=20,
                                edge_label_index=edge_label_index,
                                edge_label=edge_label, directed=directed,
                                shuffle=True)

    assert str(loader) == 'LinkNeighborLoader()'
    assert len(loader) == 1000 / 20

    for batch in loader:
        assert isinstance(batch, Data)

        assert len(batch) == 5 if labeled else 4
        assert batch.x.size(0) <= 100
        assert batch.x.min() >= 0 and batch.x.max() < 100
        assert batch.edge_index.min() >= 0
        assert batch.edge_index.max() < batch.num_nodes
        assert batch.edge_attr.min() >= 0
        assert batch.edge_attr.max() < 500

        if not labeled:
            return

        # Assert positive samples were present in the original graph:
        edge_index = unique_edge_pairs(batch.edge_index)
        edge_label_index = batch.edge_label_index[:, batch.edge_label == 1]
        edge_label_index = unique_edge_pairs(edge_label_index)
        assert len(edge_index | edge_label_index) == len(edge_index)

        # Assert negative samples were not present in the original graph:
        edge_index = unique_edge_pairs(batch.edge_index)
        edge_label_index = batch.edge_label_index[:, batch.edge_label == 0]
        edge_label_index = unique_edge_pairs(edge_label_index)
        assert len(edge_index & edge_label_index) == 0


@pytest.mark.parametrize('directed', [True, False])
@pytest.mark.parametrize('labeled', [True, False])
@pytest.mark.parametrize('subset', [True, False])
def test_heterogeneous_link_neighbor_loader(directed, labeled, subset):
    torch.manual_seed(12345)

    data = HeteroData()

    data['paper'].x = torch.arange(100)
    data['author'].x = torch.arange(100, 300)

    data['paper', 'paper'].edge_index = get_edge_index(100, 100, 500)
    data['paper', 'paper'].edge_attr = torch.arange(500)
    data['paper', 'author'].edge_index = get_edge_index(100, 200, 1000)
    data['paper', 'author'].edge_attr = torch.arange(500, 1500)
    data['author', 'paper'].edge_index = get_edge_index(200, 100, 1000)
    data['author', 'paper'].edge_attr = torch.arange(1500, 2500)

    edge_label = torch.ones(1000)

    edge_label_index = ('paper', 'to', 'author')
    if subset:
        edge_label_index = (edge_label_index, get_edge_index(100, 200, 100))

    edge_label = None
    if labeled:
        edge_label = torch.ones(100 if subset else 1000)

    loader = LinkNeighborLoader(data, num_neighbors=[-1] * 2,
                                edge_label_index=edge_label_index,
                                edge_label=edge_label, batch_size=20,
                                directed=directed, shuffle=True)

    assert str(loader) == 'LinkNeighborLoader()'
    assert len(loader) == 5 if subset else 50

    for batch in loader:
        assert isinstance(batch, HeteroData)
        assert len(batch) == 5 if labeled else 4

        assert batch['paper'].x.size(0) <= 100
        assert batch['paper'].x.min() >= 0 and batch['paper'].x.max() < 100
        assert batch['author'].x.size(0) <= 200
        assert batch['author'].x.min() >= 100 and batch['author'].x.max() < 300

        assert batch['paper', 'to', 'paper'].edge_attr.min() >= 0
        assert batch['paper', 'to', 'paper'].edge_attr.min() < 500
        assert batch['paper', 'to', 'author'].edge_attr.min() >= 500
        assert batch['paper', 'to', 'author'].edge_attr.min() < 1500
        assert batch['author', 'to', 'paper'].edge_attr.min() >= 1500
        assert batch['author', 'to', 'paper'].edge_attr.min() < 2500

        # Assert positive samples were present in the original graph:
        if not subset:
            edge_index = unique_edge_pairs(batch['paper', 'author'].edge_index)
            edge_label_index = batch['paper', 'author'].edge_label_index
            edge_label_index = unique_edge_pairs(edge_label_index)
            assert len(edge_index | edge_label_index) == len(edge_index)


@pytest.mark.parametrize('directed', [True, False])
def test_heterogeneous_link_neighbor_loader_loop(directed):
    torch.manual_seed(12345)

    data = HeteroData()

    data['paper'].x = torch.arange(100)
    data['author'].x = torch.arange(100, 300)

    data['paper', 'paper'].edge_index = get_edge_index(100, 100, 500)
    data['paper', 'paper'].edge_attr = torch.arange(500)
    data['paper', 'author'].edge_index = get_edge_index(100, 200, 1000)
    data['paper', 'author'].edge_attr = torch.arange(500, 1500)
    data['author', 'paper'].edge_index = get_edge_index(200, 100, 1000)
    data['author', 'paper'].edge_attr = torch.arange(1500, 2500)

    loader = LinkNeighborLoader(data, num_neighbors=[-1] * 2,
                                edge_label_index=("paper", "to", "paper"),
                                batch_size=20, directed=directed, shuffle=True)

    assert str(loader) == 'LinkNeighborLoader()'
    assert len(loader) == 25

    for batch in loader:
        assert isinstance(batch, HeteroData)
        assert len(batch) == 4

        assert batch['paper'].x.size(0) <= 100
        assert batch['paper'].x.min() >= 0 and batch['paper'].x.max() < 100
        assert batch['author'].x.size(0) <= 200
        assert batch['author'].x.min() >= 100 and batch['author'].x.max() < 300

        assert batch['paper', 'to', 'paper'].edge_attr.min() >= 0
        assert batch['paper', 'to', 'paper'].edge_attr.min() < 500
        assert batch['paper', 'to', 'author'].edge_attr.min() >= 500
        assert batch['paper', 'to', 'author'].edge_attr.min() < 1500
        assert batch['author', 'to', 'paper'].edge_attr.min() >= 1500
        assert batch['author', 'to', 'paper'].edge_attr.min() < 2500
