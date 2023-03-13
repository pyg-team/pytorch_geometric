import pytest
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.testing import (
    MyFeatureStore,
    MyGraphStore,
    get_random_edge_index,
    withPackage,
)


def unique_edge_pairs(edge_index):
    return set(map(tuple, edge_index.t().tolist()))


@pytest.mark.parametrize('directed', [True])  # TODO re-enable undirected mode
@pytest.mark.parametrize('neg_sampling_ratio', [None, 1.0])
def test_homo_link_neighbor_loader_basic(directed, neg_sampling_ratio):
    pos_edge_index = get_random_edge_index(100, 50, 500)
    neg_edge_index = get_random_edge_index(100, 50, 500)
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
        edge_label=edge_label if neg_sampling_ratio is None else None,
        directed=directed,
        neg_sampling_ratio=neg_sampling_ratio,
        shuffle=True,
    )

    assert str(loader) == 'LinkNeighborLoader()'
    assert len(loader) == 1000 / 20

    for batch in loader:
        assert isinstance(batch, Data)

        assert len(batch) == 8
        assert batch.n_id.size() == (batch.num_nodes, )
        assert batch.e_id.size() == (batch.num_edges, )
        assert batch.x.size(0) <= 100
        assert batch.x.min() >= 0 and batch.x.max() < 100
        assert batch.input_id.numel() == 20
        assert batch.edge_index.min() >= 0
        assert batch.edge_index.max() < batch.num_nodes
        assert batch.edge_attr.min() >= 0
        assert batch.edge_attr.max() < 500

        if neg_sampling_ratio is None:
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
@pytest.mark.parametrize('neg_sampling_ratio', [None, 1.0])
def test_hetero_link_neighbor_loader_basic(directed, neg_sampling_ratio):
    data = HeteroData()

    data['paper'].x = torch.arange(100)
    data['author'].x = torch.arange(100, 300)

    data['paper', 'paper'].edge_index = get_random_edge_index(100, 100, 500)
    data['paper', 'paper'].edge_attr = torch.arange(500)
    data['paper', 'author'].edge_index = get_random_edge_index(100, 200, 1000)
    data['paper', 'author'].edge_attr = torch.arange(500, 1500)
    data['author', 'paper'].edge_index = get_random_edge_index(200, 100, 1000)
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
        assert len(batch) == 7 + (1 if neg_sampling_ratio is not None else 0)
        if neg_sampling_ratio is None:
            # Assert only positive samples are present in the original graph:
            edge_index = unique_edge_pairs(batch['paper', 'author'].edge_index)
            edge_label_index = batch['paper', 'author'].edge_label_index
            edge_label_index = unique_edge_pairs(edge_label_index)
            assert len(edge_index | edge_label_index) == len(edge_index)

        else:
            assert batch['paper', 'author'].edge_label_index.size(1) == 40
            assert torch.all(batch['paper', 'author'].edge_label[:20] == 1)
            assert torch.all(batch['paper', 'author'].edge_label[20:] == 0)


@pytest.mark.parametrize('directed', [True])  # TODO re-enable undirected mode
def test_hetero_link_neighbor_loader_loop(directed):
    data = HeteroData()

    data['paper'].x = torch.arange(100)
    data['author'].x = torch.arange(100, 300)

    data['paper', 'paper'].edge_index = get_random_edge_index(100, 100, 500)
    data['paper', 'author'].edge_index = get_random_edge_index(100, 200, 1000)
    data['author', 'paper'].edge_index = get_random_edge_index(200, 100, 1000)

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
    edge_index = get_random_edge_index(100, 100, 500)
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
def test_temporal_hetero_link_neighbor_loader():
    data = HeteroData()

    data['paper'].x = torch.arange(100)
    data['paper'].time = torch.arange(data['paper'].num_nodes) - 200
    data['author'].x = torch.arange(100, 300)
    data['author'].time = torch.arange(data['author'].num_nodes)

    data['paper', 'paper'].edge_index = get_random_edge_index(100, 100, 500)
    data['paper', 'author'].edge_index = get_random_edge_index(100, 200, 1000)
    data['author', 'paper'].edge_index = get_random_edge_index(200, 100, 1000)

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
        assert int(batch['paper'].batch.max()) + 1 == 32

        author_max = batch['author'].time.max()
        edge_max = batch['paper', 'paper'].edge_label_time.max()
        assert edge_max >= author_max
        author_min = batch['author'].time.min()
        edge_min = batch['paper', 'paper'].edge_label_time.min()
        assert edge_min >= author_min


@pytest.mark.parametrize('FeatureStore', [MyFeatureStore, HeteroData])
@pytest.mark.parametrize('GraphStore', [MyGraphStore, HeteroData])
def test_custom_hetero_link_neighbor_loader(FeatureStore, GraphStore):
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
    edge_index = get_random_edge_index(100, 100, 500)
    data['paper', 'to', 'paper'].edge_index = edge_index
    graph_store.put_edge_index(edge_index=(edge_index[0], edge_index[1]),
                               edge_type=('paper', 'to', 'paper'),
                               layout='coo', size=(100, 100))

    edge_index = get_random_edge_index(100, 200, 1000)
    data['paper', 'to', 'author'].edge_index = edge_index
    graph_store.put_edge_index(edge_index=(edge_index[0], edge_index[1]),
                               edge_type=('paper', 'to', 'author'),
                               layout='coo', size=(100, 200))

    edge_index = get_random_edge_index(200, 100, 1000)
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
    )

    loader2 = LinkNeighborLoader(
        (feature_store, graph_store),
        num_neighbors=[-1] * 2,
        edge_label_index=('paper', 'to', 'author'),
        batch_size=20,
        directed=True,
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


def test_homo_link_neighbor_loader_no_edges():
    loader = LinkNeighborLoader(
        Data(num_nodes=100),
        num_neighbors=[],
        batch_size=20,
        edge_label_index=get_random_edge_index(100, 100, 100),
    )

    for batch in loader:
        assert isinstance(batch, Data)
        assert len(batch) == 5
        assert batch.input_id.numel() == 20
        assert batch.edge_label_index.size(1) == 20
        assert batch.num_nodes == batch.edge_label_index.unique().numel()


def test_hetero_link_neighbor_loader_no_edges():
    loader = LinkNeighborLoader(
        HeteroData(paper=dict(num_nodes=100)),
        num_neighbors=[],
        edge_label_index=(
            ('paper', 'paper'),
            get_random_edge_index(100, 100, 100),
        ),
        batch_size=20,
    )

    for batch in loader:
        assert isinstance(batch, HeteroData)
        assert len(batch) == 4
        assert batch['paper', 'paper'].input_id.numel() == 20
        assert batch['paper', 'paper'].edge_label_index.size(1) == 20
        assert batch['paper'].num_nodes == batch[
            'paper', 'paper'].edge_label_index.unique().numel()


@withPackage('pyg_lib')
@pytest.mark.parametrize('disjoint', [False, True])
@pytest.mark.parametrize('temporal', [False, True])
@pytest.mark.parametrize('amount', [1, 2])
def test_homo_link_neighbor_loader_triplet(disjoint, temporal, amount):
    if not disjoint and temporal:
        return

    data = Data()
    data.x = torch.arange(100)
    data.edge_index = get_random_edge_index(100, 100, 400)
    data.edge_label_index = get_random_edge_index(100, 100, 500)
    data.edge_attr = torch.arange(data.num_edges)

    time_attr = edge_label_time = None
    if temporal:
        time_attr = 'time'
        data.time = torch.arange(data.num_nodes)

        edge_label_time = torch.max(data.time[data.edge_label_index[0]],
                                    data.time[data.edge_label_index[1]])
        edge_label_time = edge_label_time + 50

    batch_size = 20
    loader = LinkNeighborLoader(
        data,
        num_neighbors=[-1] * 2,
        batch_size=batch_size,
        edge_label_index=data.edge_label_index,
        edge_label_time=edge_label_time,
        time_attr=time_attr,
        directed=True,
        disjoint=disjoint,
        neg_sampling=dict(mode='triplet', amount=amount),
        shuffle=True,
    )

    assert str(loader) == 'LinkNeighborLoader()'
    assert len(loader) == 500 / batch_size

    for batch in loader:
        assert isinstance(batch, Data)
        num_elems = 9 + (1 if disjoint else 0) + (2 if temporal else 0)
        assert len(batch) == num_elems

        # Check that `src_index` and `dst_pos_index` point to valid edges:
        assert torch.equal(batch.x[batch.src_index],
                           data.edge_label_index[0, batch.input_id])
        assert torch.equal(batch.x[batch.dst_pos_index],
                           data.edge_label_index[1, batch.input_id])

        # Check that `dst_neg_index` points to valid nodes in the batch:
        if amount == 1:
            assert batch.dst_neg_index.size() == (batch_size, )
        else:
            assert batch.dst_neg_index.size() == (batch_size, amount)
        assert batch.dst_neg_index.min() >= 0
        assert batch.dst_neg_index.max() < batch.num_nodes

        if disjoint:
            # In disjoint mode, seed nodes should always be placed first:
            assert batch.src_index.min() == 0
            assert batch.src_index.max() == batch_size - 1

            assert batch.dst_pos_index.min() == batch_size
            assert batch.dst_pos_index.max() == 2 * batch_size - 1

            assert batch.dst_neg_index.min() == 2 * batch_size
            max_seed_nodes = 2 * batch_size + batch_size * amount
            assert batch.dst_neg_index.max() == max_seed_nodes - 1

            assert batch.batch.min() == 0
            assert batch.batch.max() == batch_size - 1

            # Check that `batch` is always increasing:
            for i in range(0, max_seed_nodes, batch_size):
                batch_vector = batch.batch[i:i + batch_size]
                assert torch.equal(batch_vector, torch.arange(batch_size))

        if temporal:
            for i in range(batch_size):
                assert batch.time[batch.batch == i].max() <= batch.seed_time[i]


@withPackage('pyg_lib')
@pytest.mark.parametrize('disjoint', [False, True])
@pytest.mark.parametrize('temporal', [False, True])
@pytest.mark.parametrize('amount', [1, 2])
def test_hetero_link_neighbor_loader_triplet(disjoint, temporal, amount):
    if not disjoint and temporal:
        return

    data = HeteroData()

    data['paper'].x = torch.arange(100)
    data['author'].x = torch.arange(100, 300)

    data['paper', 'paper'].edge_index = get_random_edge_index(100, 100, 400)
    edge_label_index = get_random_edge_index(100, 100, 500)
    data['paper', 'paper'].edge_label_index = edge_label_index
    data['paper', 'author'].edge_index = get_random_edge_index(100, 200, 1000)
    data['author', 'paper'].edge_index = get_random_edge_index(200, 100, 1000)

    time_attr = edge_label_time = None
    if temporal:
        time_attr = 'time'
        data['paper'].time = torch.arange(data['paper'].num_nodes)
        data['author'].time = torch.arange(data['author'].num_nodes)

        edge_label_time = torch.max(
            data['paper'].time[data['paper', 'paper'].edge_label_index[0]],
            data['paper'].time[data['paper', 'paper'].edge_label_index[1]],
        )
        edge_label_time = edge_label_time + 50

    weight = torch.rand(data['paper'].num_nodes) if not temporal else None

    batch_size = 20
    index = (('paper', 'paper'), data['paper', 'paper'].edge_label_index)
    loader = LinkNeighborLoader(
        data,
        num_neighbors=[-1] * 2,
        batch_size=batch_size,
        edge_label_index=index,
        edge_label_time=edge_label_time,
        time_attr=time_attr,
        directed=True,
        disjoint=disjoint,
        neg_sampling=dict(mode='triplet', amount=amount, weight=weight),
        shuffle=True,
    )

    assert str(loader) == 'LinkNeighborLoader()'
    assert len(loader) == 500 / batch_size

    for batch in loader:
        assert isinstance(batch, HeteroData)
        num_elems = 8 + (1 if disjoint else 0) + (2 if temporal else 0)
        assert len(batch) == num_elems

        node_store = batch['paper']
        edge_store = batch['paper', 'paper']

        # Check that `src_index` and `dst_pos_index` point to valid edges:
        assert torch.equal(
            node_store.x[node_store.src_index],
            data['paper', 'paper'].edge_label_index[0, edge_store.input_id])
        assert torch.equal(
            node_store.x[node_store.dst_pos_index],
            data['paper', 'paper'].edge_label_index[1, edge_store.input_id])

        # Check that `dst_neg_index` points to valid nodes in the batch:
        if amount == 1:
            assert node_store.dst_neg_index.size() == (batch_size, )
        else:
            assert node_store.dst_neg_index.size() == (batch_size, amount)
        assert node_store.dst_neg_index.min() >= 0
        assert node_store.dst_neg_index.max() < node_store.num_nodes

        if disjoint:
            # In disjoint mode, seed nodes should always be placed first:
            assert node_store.src_index.min() == 0
            assert node_store.src_index.max() == batch_size - 1

            assert node_store.dst_pos_index.min() == batch_size
            assert node_store.dst_pos_index.max() == 2 * batch_size - 1

            assert node_store.dst_neg_index.min() == 2 * batch_size
            max_seed_nodes = 2 * batch_size + batch_size * amount
            assert node_store.dst_neg_index.max() == max_seed_nodes - 1

            assert node_store.batch.min() == 0
            assert node_store.batch.max() == batch_size - 1

            # Check that `batch` is always increasing:
            for i in range(0, max_seed_nodes, batch_size):
                batch_vector = node_store.batch[i:i + batch_size]
                assert torch.equal(batch_vector, torch.arange(batch_size))

        if temporal:
            for i in range(batch_size):
                assert (node_store.time[node_store.batch == i].max() <=
                        node_store.seed_time[i])
