import pytest
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.testing import (
    MyFeatureStore,
    MyGraphStore,
    get_random_edge_index,
    onlyNeighborSampler,
    withCUDA,
    withPackage,
)


def unique_edge_pairs(edge_index):
    return set(map(tuple, edge_index.t().tolist()))


@withCUDA
@onlyNeighborSampler
@pytest.mark.parametrize('subgraph_type', ['directional', 'bidirectional'])
@pytest.mark.parametrize('neg_sampling_ratio', [None, 1.0])
@pytest.mark.parametrize('filter_per_worker', [None, True, False])
def test_homo_link_neighbor_loader_basic(device, subgraph_type,
                                         neg_sampling_ratio,
                                         filter_per_worker):
    pos_edge_index = get_random_edge_index(50, 50, 500, device=device)
    neg_edge_index = get_random_edge_index(50, 50, 500, device=device)
    neg_edge_index += 50

    input_edges = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([
        torch.ones(500, device=device),
        torch.zeros(500, device=device),
    ], dim=0)

    data = Data()

    data.edge_index = pos_edge_index
    data.x = torch.arange(100, device=device)
    data.edge_attr = torch.arange(500, device=device)

    loader = LinkNeighborLoader(
        data,
        num_neighbors=[-1] * 2,
        batch_size=20,
        edge_label_index=input_edges,
        edge_label=edge_label if neg_sampling_ratio is None else None,
        subgraph_type=subgraph_type,
        neg_sampling_ratio=neg_sampling_ratio,
        shuffle=True,
        filter_per_worker=filter_per_worker,
    )

    assert str(loader) == 'LinkNeighborLoader()'
    assert len(loader) == 1000 / 20

    batch = loader([0])
    assert isinstance(batch, Data)
    assert int(input_edges[0, 0]) in batch.n_id.tolist()
    assert int(input_edges[1, 0]) in batch.n_id.tolist()

    for batch in loader:
        assert isinstance(batch, Data)

        assert batch.n_id.size() == (batch.num_nodes, )
        assert batch.e_id.size() == (batch.num_edges, )
        assert batch.x.device == device
        assert batch.x.size(0) <= 100
        assert batch.x.min() >= 0 and batch.x.max() < 100
        assert batch.input_id.numel() == 20
        assert batch.edge_index.device == device
        assert batch.edge_index.min() >= 0
        assert batch.edge_index.max() < batch.num_nodes
        assert batch.edge_attr.device == device
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

        # Ensure local `edge_label_index` correctly maps to input edges.
        global_edge_label_index = batch.n_id[batch.edge_label_index]
        global_edge_label_index = (
            global_edge_label_index[:, batch.edge_label >= 1])
        global_edge_label_index = unique_edge_pairs(global_edge_label_index)
        assert (len(global_edge_label_index & unique_edge_pairs(input_edges))
                == len(global_edge_label_index))


@onlyNeighborSampler
@pytest.mark.parametrize('subgraph_type', ['directional', 'bidirectional'])
@pytest.mark.parametrize('neg_sampling_ratio', [None, 1.0])
def test_hetero_link_neighbor_loader_basic(subgraph_type, neg_sampling_ratio):
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
        subgraph_type=subgraph_type,
        neg_sampling_ratio=neg_sampling_ratio,
        shuffle=True,
    )

    assert str(loader) == 'LinkNeighborLoader()'
    assert len(loader) == 1000 / 20

    for batch in loader:
        assert isinstance(batch, HeteroData)
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


@onlyNeighborSampler
@pytest.mark.parametrize('subgraph_type', ['directional', 'bidirectional'])
def test_hetero_link_neighbor_loader_loop(subgraph_type):
    data = HeteroData()

    data['paper'].x = torch.arange(100)
    data['author'].x = torch.arange(100, 300)

    data['paper', 'paper'].edge_index = get_random_edge_index(100, 100, 500)
    data['paper', 'author'].edge_index = get_random_edge_index(100, 200, 1000)
    data['author', 'paper'].edge_index = get_random_edge_index(200, 100, 1000)

    loader = LinkNeighborLoader(
        data,
        num_neighbors=[-1] * 2,
        edge_label_index=('paper', 'paper'),
        batch_size=20,
        subgraph_type=subgraph_type,
    )

    for batch in loader:
        assert batch['paper'].x.size(0) <= 100
        assert batch['paper'].x.min() >= 0 and batch['paper'].x.max() < 100

        # Assert positive samples are present in the original graph:
        edge_index = unique_edge_pairs(batch['paper', 'paper'].edge_index)
        edge_label_index = batch['paper', 'paper'].edge_label_index
        edge_label_index = unique_edge_pairs(edge_label_index)
        assert len(edge_index | edge_label_index) == len(edge_index)


@onlyNeighborSampler
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
@pytest.mark.parametrize('batch_size', [1])
def test_temporal_homo_link_neighbor_loader(batch_size):
    data = Data(
        x=torch.randn(10, 5),
        edge_index=torch.randint(0, 10, (2, 123)),
        time=torch.arange(10),
    )

    # Ensure that nodes exist at the time of the `edge_label_time`:
    edge_label_time = torch.max(
        data.time[data.edge_index[0]],
        data.time[data.edge_index[1]],
    )

    loader = LinkNeighborLoader(
        data,
        num_neighbors=[-1],
        time_attr='time',
        edge_label=torch.ones(data.num_edges),
        edge_label_time=edge_label_time,
        batch_size=batch_size,
        shuffle=True,
    )

    for batch in loader:
        assert batch.edge_label_index.size() == (2, batch_size)
        assert batch.edge_label_time.size() == (batch_size, )
        assert batch.edge_label.size() == (batch_size, )
        assert torch.all(batch.time <= batch.edge_label_time)


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


@onlyNeighborSampler
def test_custom_hetero_link_neighbor_loader():
    data = HeteroData()
    feature_store = MyFeatureStore()
    graph_store = MyGraphStore()

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
    )

    loader2 = LinkNeighborLoader(
        (feature_store, graph_store),
        num_neighbors=[-1] * 2,
        edge_label_index=('paper', 'to', 'author'),
        batch_size=20,
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


@onlyNeighborSampler
def test_homo_link_neighbor_loader_no_edges():
    loader = LinkNeighborLoader(
        Data(num_nodes=100),
        num_neighbors=[],
        batch_size=20,
        edge_label_index=get_random_edge_index(100, 100, 100),
    )

    for batch in loader:
        assert isinstance(batch, Data)
        assert batch.input_id.numel() == 20
        assert batch.edge_label_index.size(1) == 20
        assert batch.num_nodes == batch.edge_label_index.unique().numel()


@onlyNeighborSampler
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
        disjoint=disjoint,
        neg_sampling=dict(mode='triplet', amount=amount),
        shuffle=True,
    )

    assert str(loader) == 'LinkNeighborLoader()'
    assert len(loader) == 500 / batch_size

    for batch in loader:
        assert isinstance(batch, Data)

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
        disjoint=disjoint,
        neg_sampling=dict(mode='triplet', amount=amount, weight=weight),
        shuffle=True,
    )

    assert str(loader) == 'LinkNeighborLoader()'
    assert len(loader) == 500 / batch_size

    for batch in loader:
        assert isinstance(batch, HeteroData)

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
                assert (node_store.time[node_store.batch == i].max()
                        <= node_store.seed_time[i])
