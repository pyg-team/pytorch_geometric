import pytest
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.testing import withRegisteredOp


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.long)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.long)
    return torch.stack([row, col], dim=0)


def unique_edge_pairs(edge_index):
    return set(map(tuple, edge_index.t().tolist()))


@pytest.mark.parametrize('directed', [True, False])
@pytest.mark.parametrize('neg_sampling_ratio', [0.0, 1.0])
def test_homogeneous_link_neighbor_loader(directed, neg_sampling_ratio):
    torch.manual_seed(12345)

    pos_edge_index = get_edge_index(100, 50, 500)
    neg_edge_index = get_edge_index(100, 50, 500)
    neg_edge_index[1, :] += 50

    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([torch.ones(500), torch.zeros(500)], dim=0)

    data = Data()

    data.edge_index = pos_edge_index
    data.x = torch.arange(100)
    data.edge_attr = torch.arange(500)

    loader = LinkNeighborLoader(
        data,
        num_neighbors=[-1] * 2,
        batch_size=20,
        edge_label_index=edge_label_index,
        edge_label=edge_label if neg_sampling_ratio == 0.0 else None,
        directed=directed,
        neg_sampling_ratio=neg_sampling_ratio,
        shuffle=True,
    )

    assert str(loader) == 'LinkNeighborLoader()'
    assert len(loader) == 1000 / 20

    for batch in loader:
        assert isinstance(batch, Data)

        assert len(batch) == 5
        assert batch.x.size(0) <= 100
        assert batch.x.min() >= 0 and batch.x.max() < 100
        assert batch.edge_index.min() >= 0
        assert batch.edge_index.max() < batch.num_nodes
        assert batch.edge_attr.min() >= 0
        assert batch.edge_attr.max() < 500

        if neg_sampling_ratio == 0.0:
            assert batch.edge_label_index.size(1) == 20

            # Assert positive samples are present in the original graph:
            edge_index = unique_edge_pairs(batch.edge_index)
            edge_label_index = batch.edge_label_index[:, batch.edge_label == 1]
            edge_label_index = unique_edge_pairs(edge_label_index)
            assert len(edge_index | edge_label_index) == len(edge_index)

            # Assert negative samples are not present in the original graph:
            edge_index = unique_edge_pairs(batch.edge_index)
            edge_label_index = batch.edge_label_index[:, batch.edge_label == 0]
            edge_label_index = unique_edge_pairs(edge_label_index)
            assert len(edge_index & edge_label_index) == 0

        else:
            assert batch.edge_label_index.size(1) == 40
            assert torch.all(batch.edge_label[:20] == 1)
            assert torch.all(batch.edge_label[20:] == 0)


@pytest.mark.parametrize('directed', [True, False])
@pytest.mark.parametrize('neg_sampling_ratio', [0.0, 1.0])
def test_heterogeneous_link_neighbor_loader(directed, neg_sampling_ratio):
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

    loader = LinkNeighborLoader(
        data,
        num_neighbors=[-1] * 2,
        edge_label_index=('paper', 'author'),
        batch_size=20,
        directed=directed,
        neg_sampling_ratio=neg_sampling_ratio,
        shuffle=True,
    )

    assert str(loader) == 'LinkNeighborLoader()'
    assert len(loader) == 1000 / 20

    for batch in loader:
        assert isinstance(batch, HeteroData)

        if neg_sampling_ratio == 0.0:
            assert len(batch) == 4

            # Assert positive samples are present in the original graph:
            edge_index = unique_edge_pairs(batch['paper', 'author'].edge_index)
            edge_label_index = batch['paper', 'author'].edge_label_index
            edge_label_index = unique_edge_pairs(edge_label_index)
            assert len(edge_index | edge_label_index) == len(edge_index)

        else:
            assert len(batch) == 5

            assert batch['paper', 'author'].edge_label_index.size(1) == 40
            assert torch.all(batch['paper', 'author'].edge_label[:20] == 1)
            assert torch.all(batch['paper', 'author'].edge_label[20:] == 0)


@pytest.mark.parametrize('directed', [True, False])
def test_heterogeneous_link_neighbor_loader_loop(directed):
    torch.manual_seed(12345)

    data = HeteroData()

    data['paper'].x = torch.arange(100)
    data['author'].x = torch.arange(100, 300)

    data['paper', 'paper'].edge_index = get_edge_index(100, 100, 500)
    data['paper', 'author'].edge_index = get_edge_index(100, 200, 1000)
    data['author', 'paper'].edge_index = get_edge_index(200, 100, 1000)

    loader = LinkNeighborLoader(data, num_neighbors=[-1] * 2,
                                edge_label_index=('paper', 'paper'),
                                batch_size=20, directed=directed)

    for batch in loader:
        assert batch['paper'].x.size(0) <= 100
        assert batch['paper'].x.min() >= 0 and batch['paper'].x.max() < 100

        # Assert positive samples are present in the original graph:
        edge_index = unique_edge_pairs(batch['paper', 'paper'].edge_index)
        edge_label_index = batch['paper', 'paper'].edge_label_index
        edge_label_index = unique_edge_pairs(edge_label_index)
        assert len(edge_index | edge_label_index) == len(edge_index)


def test_link_neighbor_loader_edge_label():
    torch.manual_seed(12345)

    edge_index = get_edge_index(100, 100, 500)
    data = Data(edge_index=edge_index, x=torch.arange(100))

    loader = LinkNeighborLoader(
        data,
        num_neighbors=[-1] * 2,
        batch_size=10,
        neg_sampling_ratio=1.0,
    )

    for batch in loader:
        assert batch.edge_label.dtype == torch.float
        assert torch.all(batch.edge_label[:10] == 1.0)
        assert torch.all(batch.edge_label[10:] == 0.0)

    loader = LinkNeighborLoader(
        data,
        num_neighbors=[-1] * 2,
        batch_size=10,
        edge_label=torch.ones(500, dtype=torch.long),
        neg_sampling_ratio=1.0,
    )

    for batch in loader:
        assert batch.edge_label.dtype == torch.long
        assert torch.all(batch.edge_label[:10] == 2)
        assert torch.all(batch.edge_label[10:] == 0)


@withRegisteredOp('torch_sparse.hetero_temporal_neighbor_sample')
def test_temporal_heterogeneous_link_neighbor_loader():
    torch.manual_seed(12345)

    data = HeteroData()

    data['paper'].x = torch.arange(100)
    data['paper'].time = torch.arange(data['paper'].num_nodes)
    data['author'].x = torch.arange(100, 300)

    data['paper', 'paper'].edge_index = get_edge_index(100, 100, 500)
    data['paper', 'author'].edge_index = get_edge_index(100, 200, 1000)
    data['author', 'paper'].edge_index = get_edge_index(200, 100, 1000)

    loader = LinkNeighborLoader(data, num_neighbors=[-1] * 2,
                                edge_label_index=('paper', 'paper'),
                                batch_size=1, time_attr='time')

    for batch in loader:
        mask = batch['paper'].time[0] >= batch['paper'].time[1:]
        assert torch.all(mask)
