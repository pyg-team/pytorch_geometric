import pytest
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.testing import withPackage
from torch_geometric.testing.feature_store import MyFeatureStore
from torch_geometric.testing.graph_store import MyGraphStore


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.long)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.long)
    return torch.stack([row, col], dim=0)


def unique_edge_pairs(edge_index):
    return set(map(tuple, edge_index.t().tolist()))


@pytest.mark.parametrize('directed', [True])  # TODO re-enable undirected mode
@pytest.mark.parametrize('neg_sampling_ratio', [0.0, 1.0])
def test_homogeneous_link_neighbor_loader(directed, neg_sampling_ratio):
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

        assert len(batch) == 6
        assert batch.x.size(0) <= 100
        assert batch.x.min() >= 0 and batch.x.max() < 100
        assert batch.input_id.numel() == 20
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


@pytest.mark.parametrize('directed', [True])  # TODO re-enable undirected mode
@pytest.mark.parametrize('neg_sampling_ratio', [0.0, 1.0])
def test_heterogeneous_link_neighbor_loader(directed, neg_sampling_ratio):
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
        assert len(batch) == 6
        if neg_sampling_ratio == 0.0:
            # Assert only positive samples are present in the original graph:
            assert batch['paper', 'author'].edge_label.sum() == 0
            edge_index = unique_edge_pairs(batch['paper', 'author'].edge_index)
            edge_label_index = batch['paper', 'author'].edge_label_index
            edge_label_index = unique_edge_pairs(edge_label_index)
            assert len(edge_index | edge_label_index) == len(edge_index)

        else:
            assert batch['paper', 'author'].edge_label_index.size(1) == 40
            assert torch.all(batch['paper', 'author'].edge_label[:20] == 1)
            assert torch.all(batch['paper', 'author'].edge_label[20:] == 0)


@pytest.mark.parametrize('directed', [True])  # TODO re-enable undirected mode
def test_heterogeneous_link_neighbor_loader_loop(directed):
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
        assert torch.all(batch.edge_label[:10] == 1)
        assert torch.all(batch.edge_label[10:] == 0)


@withPackage('pyg_lib')
def test_temporal_heterogeneous_link_neighbor_loader():
    data = HeteroData()

    data['paper'].x = torch.arange(100)
    data['paper'].time = torch.arange(data['paper'].num_nodes) - 200
    data['author'].x = torch.arange(100, 300)
    data['author'].time = torch.arange(data['author'].num_nodes)

    data['paper', 'paper'].edge_index = get_edge_index(100, 100, 500)
    data['paper', 'author'].edge_index = get_edge_index(100, 200, 1000)
    data['author', 'paper'].edge_index = get_edge_index(200, 100, 1000)

    with pytest.raises(ValueError, match=r"'edge_label_time' is not set"):
        loader = LinkNeighborLoader(
            data,
            num_neighbors=[-1] * 2,
            edge_label_index=('paper', 'paper'),
            batch_size=32,
            time_attr='time',
        )

    # With edge_time:
    edge_time = torch.arange(data['paper', 'paper'].edge_index.size(1))
    loader = LinkNeighborLoader(
        data,
        num_neighbors=[-1] * 2,
        edge_label_index=('paper', 'paper'),
        edge_label_time=edge_time,
        batch_size=32,
        time_attr='time',
        neg_sampling_ratio=0.5,
        drop_last=True,
    )
    for batch in loader:
        # Check if each seed edge has a different batch:
        assert int(batch['paper'].batch.max()) + 1 == 32 + 16
        # Check if each seed edge has a different source and dstination node:
        assert batch['paper'].num_nodes >= 2 * (32 + 16)

        author_max = batch['author'].time.max()
        edge_max = batch['paper', 'paper'].edge_label_time.max()
        assert edge_max >= author_max
        author_min = batch['author'].time.min()
        edge_min = batch['paper', 'paper'].edge_label_time.min()
        assert edge_min >= author_min


@pytest.mark.parametrize('FeatureStore', [MyFeatureStore, HeteroData])
@pytest.mark.parametrize('GraphStore', [MyGraphStore, HeteroData])
def test_custom_heterogeneous_link_neighbor_loader(FeatureStore, GraphStore):
    data = HeteroData()
    feature_store = FeatureStore()
    graph_store = GraphStore()

    # Set up node features:
    x = torch.arange(100)
    data['paper'].x = x
    feature_store.put_tensor(x, group_name='paper', attr_name='x', index=None)

    x = torch.arange(100, 300)
    data['author'].x = x
    feature_store.put_tensor(x, group_name='author', attr_name='x', index=None)

    # Set up edge indices (GraphStore does not support `edge_attr` at the
    # moment):
    edge_index = get_edge_index(100, 100, 500)
    data['paper', 'to', 'paper'].edge_index = edge_index
    graph_store.put_edge_index(edge_index=(edge_index[0], edge_index[1]),
                               edge_type=('paper', 'to', 'paper'),
                               layout='coo', size=(100, 100))

    edge_index = get_edge_index(100, 200, 1000)
    data['paper', 'to', 'author'].edge_index = edge_index
    graph_store.put_edge_index(edge_index=(edge_index[0], edge_index[1]),
                               edge_type=('paper', 'to', 'author'),
                               layout='coo', size=(100, 200))

    edge_index = get_edge_index(200, 100, 1000)
    data['author', 'to', 'paper'].edge_index = edge_index
    graph_store.put_edge_index(edge_index=(edge_index[0], edge_index[1]),
                               edge_type=('author', 'to', 'paper'),
                               layout='coo', size=(200, 100))

    loader1 = LinkNeighborLoader(
        data,
        num_neighbors=[-1] * 2,
        edge_label_index=('paper', 'to', 'author'),
        batch_size=20,
        directed=True,
        neg_sampling_ratio=0,
    )

    loader2 = LinkNeighborLoader(
        (feature_store, graph_store),
        num_neighbors=[-1] * 2,
        edge_label_index=('paper', 'to', 'author'),
        batch_size=20,
        directed=True,
        neg_sampling_ratio=0,
    )

    assert str(loader1) == str(loader2)

    for (batch1, batch2) in zip(loader1, loader2):
        # Mapped indices of neighbors may be differently sorted:
        assert torch.allclose(batch1['paper'].x.sort()[0],
                              batch2['paper'].x.sort()[0])
        assert torch.allclose(batch1['author'].x.sort()[0],
                              batch2['author'].x.sort()[0])

        # Assert that edge indices have the same size:
        assert (batch1['paper', 'to', 'paper'].edge_index.size() == batch1[
            'paper', 'to', 'paper'].edge_index.size())
        assert (batch1['paper', 'to', 'author'].edge_index.size() == batch1[
            'paper', 'to', 'author'].edge_index.size())
        assert (batch1['author', 'to', 'paper'].edge_index.size() == batch1[
            'author', 'to', 'paper'].edge_index.size())


def test_homogeneous_link_neighbor_loader_no_edges():
    loader = LinkNeighborLoader(
        Data(num_nodes=100),
        num_neighbors=[],
        batch_size=20,
        edge_label_index=get_edge_index(100, 100, 100),
    )

    for batch in loader:
        assert isinstance(batch, Data)
        assert len(batch) == 4
        assert batch.input_id.numel() == 20
        assert batch.num_nodes <= 40
        assert batch.edge_label_index.size(1) == 20
        assert batch.num_nodes == batch.edge_label_index.unique().numel()


def test_heterogeneous_link_neighbor_loader_no_edges():
    loader = LinkNeighborLoader(
        HeteroData(paper=dict(num_nodes=100)),
        num_neighbors=[],
        edge_label_index=(('paper', 'paper'), get_edge_index(100, 100, 100)),
        batch_size=20,
    )

    for batch in loader:
        assert isinstance(batch, HeteroData)
        assert len(batch) == 4
        assert batch['paper'].num_nodes <= 40
        assert batch['paper', 'paper'].input_id.numel() == 20
        assert batch['paper', 'paper'].edge_label_index.size(1) == 20
        assert batch['paper'].num_nodes == batch[
            'paper', 'paper'].edge_label_index.unique().numel()
