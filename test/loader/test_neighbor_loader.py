import os.path as osp

import numpy as np
import pytest
import torch

import torch_geometric.typing
from torch_geometric import EdgeIndex
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GraphConv, to_hetero
from torch_geometric.sampler.base import SubgraphType
from torch_geometric.testing import (
    MyFeatureStore,
    MyGraphStore,
    get_random_edge_index,
    get_random_tensor_frame,
    onlyLinux,
    onlyNeighborSampler,
    onlyOnline,
    withCUDA,
    withPackage,
)
from torch_geometric.typing import (
    WITH_EDGE_TIME_NEIGHBOR_SAMPLE,
    WITH_PYG_LIB,
    WITH_TORCH_SPARSE,
    WITH_WEIGHTED_NEIGHBOR_SAMPLE,
    TensorFrame,
)
from torch_geometric.utils import (
    is_undirected,
    sort_edge_index,
    to_torch_csr_tensor,
    to_undirected,
)

DTYPES = [
    pytest.param(torch.int64, id='int64'),
    pytest.param(torch.int32, id='int32'),
]

SUBGRAPH_TYPES = [
    pytest.param(SubgraphType.directional, id='directional'),
    pytest.param(SubgraphType.bidirectional, id='bidirectional'),
    pytest.param(SubgraphType.induced, id='induced'),
]

FILTER_PER_WORKERS = [
    pytest.param(None, id='auto_filter'),
    pytest.param(True, id='filter_per_worker'),
    pytest.param(False, id='filter_in_main'),
]


def is_subset(subedge_index, edge_index, src_idx, dst_idx):
    num_nodes = int(edge_index.max()) + 1
    idx = num_nodes * edge_index[0] + edge_index[1]
    subidx = num_nodes * src_idx[subedge_index[0]] + dst_idx[subedge_index[1]]
    mask = torch.from_numpy(np.isin(subidx.cpu().numpy(), idx.cpu().numpy()))
    return int(mask.sum()) == mask.numel()


@withCUDA
@onlyNeighborSampler
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('subgraph_type', SUBGRAPH_TYPES)
@pytest.mark.parametrize('filter_per_worker', FILTER_PER_WORKERS)
def test_homo_neighbor_loader_basic(
    device,
    subgraph_type,
    dtype,
    filter_per_worker,
):
    if dtype != torch.int64 and not torch_geometric.typing.WITH_PT20:
        return
    induced = SubgraphType.induced
    if subgraph_type == SubgraphType.induced and not WITH_TORCH_SPARSE:
        return
    if dtype != torch.int64 and (not WITH_PYG_LIB or subgraph_type == induced):
        return

    torch.manual_seed(12345)

    data = Data()

    data.x = torch.arange(100, device=device)
    data.edge_index = get_random_edge_index(100, 100, 500, dtype, device)
    data.edge_attr = torch.arange(500, device=device)

    loader = NeighborLoader(
        data,
        num_neighbors=[5] * 2,
        batch_size=20,
        subgraph_type=subgraph_type,
        filter_per_worker=filter_per_worker,
    )

    assert str(loader) == 'NeighborLoader()'
    assert len(loader) == 5

    batch = loader([0])
    assert isinstance(batch, Data)
    assert batch.n_id[:1].tolist() == [0]

    for i, batch in enumerate(loader):
        assert isinstance(batch, Data)
        assert batch.x.device == device
        assert batch.x.size(0) <= 100
        assert batch.n_id.size() == (batch.num_nodes, )
        assert batch.input_id.numel() == batch.batch_size == 20
        assert batch.x.min() >= 0 and batch.x.max() < 100
        # TODO Re-enable once `EdgeIndex` is stable.
        assert not isinstance(batch.edge_index, EdgeIndex)
        # batch.edge_index.validate()
        # size = (batch.num_nodes, batch.num_nodes)
        # assert batch.edge_index.sparse_size() == size
        # assert batch.edge_index.sort_order == 'col'
        assert batch.edge_index.device == device
        assert batch.edge_index.min() >= 0
        assert batch.edge_index.max() < batch.num_nodes
        assert batch.edge_attr.device == device
        assert batch.edge_attr.size(0) == batch.edge_index.size(1)

        # Input nodes are always sampled first:
        assert torch.equal(
            batch.x[:batch.batch_size],
            torch.arange(i * batch.batch_size, (i + 1) * batch.batch_size,
                         device=device),
        )

        if subgraph_type != SubgraphType.bidirectional:
            assert batch.e_id.size() == (batch.num_edges, )
            assert batch.edge_attr.min() >= 0
            assert batch.edge_attr.max() < 500

            assert is_subset(
                batch.edge_index.to(torch.int64),
                data.edge_index.to(torch.int64),
                batch.x,
                batch.x,
            )


@onlyNeighborSampler
@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('subgraph_type', SUBGRAPH_TYPES)
def test_hetero_neighbor_loader_basic(subgraph_type, dtype):
    if dtype != torch.int64 and not torch_geometric.typing.WITH_PT20:
        return
    induced = SubgraphType.induced
    if subgraph_type == SubgraphType.induced and not WITH_TORCH_SPARSE:
        return
    if dtype != torch.int64 and (not WITH_PYG_LIB or subgraph_type == induced):
        return

    torch.manual_seed(12345)

    data = HeteroData()

    data['paper'].x = torch.arange(100)
    data['author'].x = torch.arange(100, 300)

    edge_index = get_random_edge_index(100, 100, 500, dtype)
    data['paper', 'paper'].edge_index = edge_index
    data['paper', 'paper'].edge_attr = torch.arange(500)
    edge_index = get_random_edge_index(100, 200, 1000, dtype)
    data['paper', 'author'].edge_index = edge_index
    data['paper', 'author'].edge_attr = torch.arange(500, 1500)
    edge_index = get_random_edge_index(200, 100, 1000, dtype)
    data['author', 'paper'].edge_index = edge_index
    data['author', 'paper'].edge_attr = torch.arange(1500, 2500)

    r1, c1 = data['paper', 'paper'].edge_index
    r2, c2 = data['paper', 'author'].edge_index + torch.tensor([[0], [100]])
    r3, c3 = data['author', 'paper'].edge_index + torch.tensor([[100], [0]])

    batch_size = 20

    with pytest.raises(ValueError, match="hops must be the same across all"):
        loader = NeighborLoader(
            data,
            num_neighbors={
                ('paper', 'to', 'paper'): [-1],
                ('paper', 'to', 'author'): [-1, -1],
                ('author', 'to', 'paper'): [-1, -1],
            },
            input_nodes='paper',
            batch_size=batch_size,
            subgraph_type=subgraph_type,
        )
        next(iter(loader))

    loader = NeighborLoader(
        data,
        num_neighbors=[10] * 2,
        input_nodes='paper',
        batch_size=batch_size,
        subgraph_type=subgraph_type,
    )

    assert str(loader) == 'NeighborLoader()'
    assert len(loader) == (100 + batch_size - 1) // batch_size

    for batch in loader:
        assert isinstance(batch, HeteroData)

        # Test node type selection:
        assert set(batch.node_types) == {'paper', 'author'}

        assert batch['paper'].n_id.size() == (batch['paper'].num_nodes, )
        assert batch['paper'].x.size(0) <= 100
        assert batch['paper'].input_id.numel() == batch_size
        assert batch['paper'].batch_size == batch_size
        assert batch['paper'].x.min() >= 0 and batch['paper'].x.max() < 100

        assert batch['author'].n_id.size() == (batch['author'].num_nodes, )
        assert batch['author'].x.size(0) <= 200
        assert batch['author'].x.min() >= 100 and batch['author'].x.max() < 300

        # Test edge type selection:
        assert set(batch.edge_types) == {('paper', 'to', 'paper'),
                                         ('paper', 'to', 'author'),
                                         ('author', 'to', 'paper')}

        for edge_type, edge_index in batch.edge_index_dict.items():
            src, _, dst = edge_type
            # TODO Re-enable once `EdgeIndex` is stable.
            assert not isinstance(edge_index, EdgeIndex)
            # edge_index.validate()
            # size = (batch[src].num_nodes, batch[dst].num_nodes)
            # assert edge_index.sparse_size() == size
            # assert edge_index.sort_order == 'col'

        row, col = batch['paper', 'paper'].edge_index
        assert row.min() >= 0 and row.max() < batch['paper'].num_nodes
        assert col.min() >= 0 and col.max() < batch['paper'].num_nodes

        if subgraph_type != SubgraphType.bidirectional:
            assert batch['paper', 'paper'].e_id.size() == (row.numel(), )
            value = batch['paper', 'paper'].edge_attr
            assert value.min() >= 0 and value.max() < 500

            assert is_subset(
                batch['paper', 'paper'].edge_index.to(torch.int64),
                data['paper', 'paper'].edge_index.to(torch.int64),
                batch['paper'].x,
                batch['paper'].x,
            )
        elif subgraph_type != SubgraphType.directional:
            assert 'e_id' not in batch['paper', 'paper']
            assert 'edge_attr' not in batch['paper', 'paper']

            assert is_undirected(batch['paper', 'paper'].edge_index)

        row, col = batch['paper', 'author'].edge_index
        assert row.min() >= 0 and row.max() < batch['paper'].num_nodes
        assert col.min() >= 0 and col.max() < batch['author'].num_nodes

        if subgraph_type != SubgraphType.bidirectional:
            assert batch['paper', 'author'].e_id.size() == (row.numel(), )
            value = batch['paper', 'author'].edge_attr
            assert value.min() >= 500 and value.max() < 1500

            assert is_subset(
                batch['paper', 'author'].edge_index.to(torch.int64),
                data['paper', 'author'].edge_index.to(torch.int64),
                batch['paper'].x,
                batch['author'].x - 100,
            )
        elif subgraph_type != SubgraphType.directional:
            assert 'e_id' not in batch['paper', 'author']
            assert 'edge_attr' not in batch['paper', 'author']

            edge_index1 = batch['paper', 'author'].edge_index
            edge_index2 = batch['author', 'paper'].edge_index
            assert torch.equal(
                edge_index1,
                sort_edge_index(edge_index2.flip([0]), sort_by_row=False),
            )

        row, col = batch['author', 'paper'].edge_index
        assert row.min() >= 0 and row.max() < batch['author'].num_nodes
        assert col.min() >= 0 and col.max() < batch['paper'].num_nodes

        if subgraph_type != SubgraphType.bidirectional:
            assert batch['author', 'paper'].e_id.size() == (row.numel(), )
            value = batch['author', 'paper'].edge_attr
            assert value.min() >= 1500 and value.max() < 2500

            assert is_subset(
                batch['author', 'paper'].edge_index.to(torch.int64),
                data['author', 'paper'].edge_index.to(torch.int64),
                batch['author'].x - 100,
                batch['paper'].x,
            )
        elif subgraph_type != SubgraphType.directional:
            assert 'e_id' not in batch['author', 'paper']
            assert 'edge_attr' not in batch['author', 'paper']

            edge_index1 = batch['author', 'paper'].edge_index
            edge_index2 = batch['paper', 'author'].edge_index
            assert torch.equal(
                edge_index1,
                sort_edge_index(edge_index2.flip([0]), sort_by_row=False),
            )

        # Test for isolated nodes (there shouldn't exist any):
        assert not batch.has_isolated_nodes()


@onlyOnline
@onlyNeighborSampler
@pytest.mark.parametrize('subgraph_type', SUBGRAPH_TYPES)
def test_homo_neighbor_loader_on_karate(get_dataset, subgraph_type):
    if subgraph_type == SubgraphType.induced and not WITH_TORCH_SPARSE:
        return
    dataset = get_dataset(name='karate')
    data = dataset[0]

    mask = data.edge_index[0] < data.edge_index[1]
    edge_index = data.edge_index[:, mask]
    edge_weight = torch.rand(edge_index.size(1))
    data.edge_index, data.edge_weight = to_undirected(edge_index, edge_weight)

    split_idx = torch.arange(5, 8)

    loader = NeighborLoader(
        data,
        num_neighbors=[-1, -1],
        batch_size=split_idx.numel(),
        input_nodes=split_idx,
        subgraph_type=subgraph_type,
    )
    assert len(loader) == 1

    batch = next(iter(loader))
    batch_size = batch.batch_size

    class GNN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = GraphConv(in_channels, hidden_channels)
            self.conv2 = GraphConv(hidden_channels, out_channels)

        def forward(self, x, edge_index, edge_weight):
            x = self.conv1(x, edge_index, edge_weight).relu()
            x = self.conv2(x, edge_index, edge_weight)
            return x

    model = GNN(dataset.num_features, 16, dataset.num_classes)

    out1 = model(data.x, data.edge_index, data.edge_weight)[split_idx]
    out2 = model(batch.x, batch.edge_index, batch.edge_weight)[:batch_size]
    assert torch.allclose(out1, out2, atol=1e-6)


@onlyOnline
@onlyNeighborSampler
@pytest.mark.parametrize('subgraph_type', SUBGRAPH_TYPES)
def test_hetero_neighbor_loader_on_karate(get_dataset, subgraph_type):
    if subgraph_type == SubgraphType.induced and not WITH_TORCH_SPARSE:
        return
    dataset = get_dataset(name='karate')
    data = dataset[0]

    hetero_data = HeteroData()
    hetero_data['v'].x = data.x
    hetero_data['v', 'v'].edge_index = data.edge_index

    split_idx = torch.arange(5, 8)

    loader = NeighborLoader(
        hetero_data,
        num_neighbors=[-1, -1],
        batch_size=split_idx.numel(),
        input_nodes=('v', split_idx),
        subgraph_type=subgraph_type,
    )
    assert len(loader) == 1

    hetero_batch = next(iter(loader))
    batch_size = hetero_batch['v'].batch_size

    class GNN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = GraphConv(in_channels, hidden_channels)
            self.conv2 = GraphConv(hidden_channels, out_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index)
            return x

    model = GNN(dataset.num_features, 16, dataset.num_classes)
    hetero_model = to_hetero(model, hetero_data.metadata())

    out1 = model(data.x, data.edge_index)[split_idx]
    out2 = hetero_model(hetero_batch.x_dict,
                        hetero_batch.edge_index_dict)['v'][:batch_size]
    assert torch.allclose(out1, out2, atol=1e-6)


@onlyOnline
@withPackage('pyg_lib')
def test_temporal_hetero_neighbor_loader_on_karate(get_dataset):
    dataset = get_dataset(name='karate')
    data = dataset[0]

    hetero_data = HeteroData()
    hetero_data['v'].x = data.x
    hetero_data['v'].time = torch.arange(data.num_nodes, 0, -1)
    hetero_data['v', 'v'].edge_index = data.edge_index

    loader = NeighborLoader(hetero_data, num_neighbors=[-1, -1],
                            input_nodes='v', time_attr='time', batch_size=1)

    for batch in loader:
        mask = batch['v'].time[0] >= batch['v'].time[1:]
        assert torch.all(mask)


@onlyNeighborSampler
def test_custom_neighbor_loader():
    # Initialize feature store, graph store, and reference:
    feature_store = MyFeatureStore()
    graph_store = MyGraphStore()

    # Set up node features:
    x = torch.arange(100, 300)
    feature_store.put_tensor(x, group_name=None, attr_name='x', index=None)

    y = torch.arange(100, 300)
    feature_store.put_tensor(y, group_name=None, attr_name='y', index=None)

    # COO:
    edge_index = get_random_edge_index(100, 100, 500, coalesce=True)
    edge_index = edge_index[:, torch.randperm(edge_index.size(1))]
    coo = (edge_index[0], edge_index[1])
    graph_store.put_edge_index(edge_index=coo, edge_type=None, layout='coo',
                               size=(100, 100))

    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=200)

    # Construct neighbor loaders:
    loader1 = NeighborLoader(data, batch_size=20,
                             input_nodes=torch.arange(100),
                             num_neighbors=[-1] * 2)

    loader2 = NeighborLoader((feature_store, graph_store), batch_size=20,
                             input_nodes=torch.arange(100),
                             num_neighbors=[-1] * 2)

    assert str(loader1) == str(loader2)
    assert len(loader1) == len(loader2)

    for batch1, batch2 in zip(loader1, loader2):
        assert len(batch1) == len(batch2)
        assert batch1.num_nodes == batch2.num_nodes
        assert batch1.num_edges == batch2.num_edges
        assert batch1.batch_size == batch2.batch_size

        # Mapped indices of neighbors may be differently sorted ...
        assert torch.allclose(batch1.x.sort()[0], batch2.x.sort()[0])
        assert torch.allclose(batch1.y.sort()[0], batch2.y.sort()[0])


@onlyNeighborSampler
def test_custom_hetero_neighbor_loader():
    # Initialize feature store, graph store, and reference:
    feature_store = MyFeatureStore()
    graph_store = MyGraphStore()
    data = HeteroData()

    # Set up node features:
    x = torch.arange(100)
    data['paper'].x = x
    feature_store.put_tensor(x, group_name='paper', attr_name='x', index=None)

    x = torch.arange(100, 300)
    data['author'].x = x
    feature_store.put_tensor(x, group_name='author', attr_name='x', index=None)

    # COO:
    edge_index = get_random_edge_index(100, 100, 500, coalesce=True)
    edge_index = edge_index[:, torch.randperm(edge_index.size(1))]
    data['paper', 'to', 'paper'].edge_index = edge_index
    coo = (edge_index[0], edge_index[1])
    graph_store.put_edge_index(edge_index=coo,
                               edge_type=('paper', 'to', 'paper'),
                               layout='coo', size=(100, 100))

    # CSR:
    edge_index = get_random_edge_index(100, 200, 1000, coalesce=True)
    data['paper', 'to', 'author'].edge_index = edge_index
    adj = to_torch_csr_tensor(edge_index, size=(100, 200))
    csr = (adj.crow_indices(), adj.col_indices())
    graph_store.put_edge_index(edge_index=csr,
                               edge_type=('paper', 'to', 'author'),
                               layout='csr', size=(100, 200))

    # CSC:
    edge_index = get_random_edge_index(200, 100, 1000, coalesce=True)
    data['author', 'to', 'paper'].edge_index = edge_index
    adj = to_torch_csr_tensor(edge_index.flip([0]), size=(100, 200))
    csc = (adj.col_indices(), adj.crow_indices())
    graph_store.put_edge_index(edge_index=csc,
                               edge_type=('author', 'to', 'paper'),
                               layout='csc', size=(200, 100))

    # COO (sorted):
    edge_index = get_random_edge_index(200, 200, 100, coalesce=True)
    edge_index = edge_index[:, edge_index[1].argsort()]
    data['author', 'to', 'author'].edge_index = edge_index
    coo = (edge_index[0], edge_index[1])
    graph_store.put_edge_index(edge_index=coo,
                               edge_type=('author', 'to', 'author'),
                               layout='coo', size=(200, 200), is_sorted=True)

    # Construct neighbor loaders:
    loader1 = NeighborLoader(data, batch_size=20,
                             input_nodes=('paper', range(100)),
                             num_neighbors=[-1] * 2)

    loader2 = NeighborLoader((feature_store, graph_store), batch_size=20,
                             input_nodes=('paper', range(100)),
                             num_neighbors=[-1] * 2)

    assert str(loader1) == str(loader2)
    assert len(loader1) == len(loader2)

    for batch1, batch2 in zip(loader1, loader2):
        # `loader2` explicitly adds `num_nodes` to the batch:
        assert len(batch1) + 1 == len(batch2)
        assert batch1['paper'].batch_size == batch2['paper'].batch_size

        # Mapped indices of neighbors may be differently sorted ...
        for node_type in data.node_types:
            assert torch.allclose(
                batch1[node_type].x.sort()[0],
                batch2[node_type].x.sort()[0],
            )

        # ... but should sample the exact same number of edges:
        for edge_type in data.edge_types:
            assert batch1[edge_type].num_edges == batch2[edge_type].num_edges


@onlyOnline
@withPackage('pyg_lib')
def test_temporal_custom_neighbor_loader_on_karate(get_dataset):
    dataset = get_dataset(name='karate')
    data = dataset[0]
    data.time = torch.arange(data.num_nodes, 0, -1)

    # Initialize feature store, graph store, and reference:
    feature_store = MyFeatureStore()
    graph_store = MyGraphStore()
    hetero_data = HeteroData()

    feature_store.put_tensor(
        data.x,
        group_name='v',
        attr_name='x',
        index=None,
    )
    hetero_data['v'].x = data.x

    feature_store.put_tensor(
        data.time,
        group_name='v',
        attr_name='time',
        index=None,
    )
    hetero_data['v'].time = data.time

    # Sort according to time in local neighborhoods:
    row, col = data.edge_index
    perm = ((col * (data.num_nodes + 1)) + data.time[row]).argsort()
    edge_index = data.edge_index[:, perm]

    graph_store.put_edge_index(
        edge_index,
        edge_type=('v', 'to', 'v'),
        layout='coo',
        is_sorted=True,
        size=(data.num_nodes, data.num_nodes),
    )
    hetero_data['v', 'to', 'v'].edge_index = data.edge_index

    loader1 = NeighborLoader(
        hetero_data,
        num_neighbors=[-1, -1],
        input_nodes='v',
        time_attr='time',
        batch_size=128,
    )

    loader2 = NeighborLoader(
        (feature_store, graph_store),
        num_neighbors=[-1, -1],
        input_nodes='v',
        time_attr='time',
        batch_size=128,
    )

    for batch1, batch2 in zip(loader1, loader2):
        assert torch.equal(batch1['v'].time, batch2['v'].time)


@withPackage('pyg_lib', 'torch_sparse')
def test_pyg_lib_and_torch_sparse_homo_equality():
    edge_index = get_random_edge_index(20, 20, 100)
    adj = to_torch_csr_tensor(edge_index.flip([0]), size=(20, 20))
    colptr, row = adj.crow_indices(), adj.col_indices()

    seed = torch.arange(10)

    sample = torch.ops.pyg.neighbor_sample
    out1 = sample(colptr, row, seed, [-1, -1], None, None, None, None, True)
    sample = torch.ops.torch_sparse.neighbor_sample
    out2 = sample(colptr, row, seed, [-1, -1], False, True)

    row1, col1, node_id1, edge_id1 = out1[:4]
    node_id2, row2, col2, edge_id2 = out2
    assert torch.equal(node_id1, node_id2)
    assert torch.equal(row1, row2)
    assert torch.equal(col1, col2)
    assert torch.equal(edge_id1, edge_id2)


@withPackage('pyg_lib', 'torch_sparse')
def test_pyg_lib_and_torch_sparse_hetero_equality():
    edge_index = get_random_edge_index(20, 10, 50)
    adj = to_torch_csr_tensor(edge_index.flip([0]), size=(10, 20))
    colptr1, row1 = adj.crow_indices(), adj.col_indices()

    edge_index = get_random_edge_index(10, 20, 50)
    adj = to_torch_csr_tensor(edge_index.flip([0]), size=(20, 10))
    colptr2, row2 = adj.crow_indices(), adj.col_indices()

    node_types = ['paper', 'author']
    edge_types = [('paper', 'to', 'author'), ('author', 'to', 'paper')]
    colptr_dict = {
        'paper__to__author': colptr1,
        'author__to__paper': colptr2,
    }
    row_dict = {
        'paper__to__author': row1,
        'author__to__paper': row2,
    }
    seed_dict = {'paper': torch.arange(1)}
    num_neighbors_dict = {
        'paper__to__author': [-1, -1],
        'author__to__paper': [-1, -1],
    }

    sample = torch.ops.pyg.hetero_neighbor_sample
    out1 = sample(node_types, edge_types, colptr_dict, row_dict, seed_dict,
                  num_neighbors_dict, None, None, None, None, True, False,
                  True, False, "uniform", True)
    sample = torch.ops.torch_sparse.hetero_neighbor_sample
    out2 = sample(node_types, edge_types, colptr_dict, row_dict, seed_dict,
                  num_neighbors_dict, 2, False, True)

    row1_dict, col1_dict, node_id1_dict, edge_id1_dict = out1[:4]
    node_id2_dict, row2_dict, col2_dict, edge_id2_dict = out2
    assert len(node_id1_dict) == len(node_id2_dict)
    for key in node_id1_dict.keys():
        assert torch.equal(node_id1_dict[key], node_id2_dict[key])
    assert len(row1_dict) == len(row2_dict)
    for key in row1_dict.keys():
        assert torch.equal(row1_dict[key], row2_dict[key])
    assert len(col1_dict) == len(col2_dict)
    for key in col1_dict.keys():
        assert torch.equal(col1_dict[key], col2_dict[key])
    assert len(edge_id1_dict) == len(edge_id2_dict)
    for key in edge_id1_dict.keys():
        assert torch.equal(edge_id1_dict[key], edge_id2_dict[key])


@onlyLinux
@onlyNeighborSampler
def test_memmap_neighbor_loader(tmp_path):
    path = osp.join(tmp_path, 'x.npy')
    x = np.memmap(path, dtype=np.float32, mode='w+', shape=(100, 32))
    x[:] = np.random.randn(100, 32)

    data = Data()
    data.x = np.memmap(path, dtype=np.float32, mode='r', shape=(100, 32))
    data.edge_index = get_random_edge_index(100, 100, 500)

    assert str(data) == 'Data(x=[100, 32], edge_index=[2, 500])'
    assert data.num_nodes == 100

    loader = NeighborLoader(data, num_neighbors=[5] * 2, batch_size=20,
                            num_workers=2)
    batch = next(iter(loader))
    assert batch.num_nodes <= 100
    assert isinstance(batch.x, torch.Tensor)
    assert batch.x.size() == (batch.num_nodes, 32)


@withPackage('pyg_lib')
def test_homo_neighbor_loader_sampled_info():
    edge_index = torch.tensor([
        [2, 3, 4, 5, 7, 7, 10, 11, 12, 13],
        [0, 1, 2, 3, 2, 3, 7, 7, 7, 7],
    ])

    data = Data(edge_index=edge_index, num_nodes=14)

    loader = NeighborLoader(
        data,
        num_neighbors=[1, 2, 4],
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(loader))

    assert batch.num_sampled_nodes == [2, 2, 3, 4]
    assert batch.num_sampled_edges == [2, 4, 4]


@withPackage('pyg_lib')
def test_hetero_neighbor_loader_sampled_info():
    edge_index = torch.tensor([
        [2, 3, 4, 5, 7, 7, 10, 11, 12, 13],
        [0, 1, 2, 3, 2, 3, 7, 7, 7, 7],
    ])

    data = HeteroData()
    data['paper'].num_nodes = data['author'].num_nodes = 14
    data['paper', 'paper'].edge_index = edge_index
    data['paper', 'author'].edge_index = edge_index
    data['author', 'paper'].edge_index = edge_index

    loader = NeighborLoader(
        data,
        num_neighbors=[1, 2, 4],
        batch_size=2,
        input_nodes='paper',
        shuffle=False,
    )
    batch = next(iter(loader))

    expected_num_sampled_nodes = {
        'paper': [2, 2, 3, 4],
        'author': [0, 2, 3, 4],
    }
    expected_num_sampled_edges = {
        ('paper', 'to', 'paper'): [2, 4, 4],
        ('paper', 'to', 'author'): [0, 4, 4],
        ('author', 'to', 'paper'): [2, 4, 4],
    }

    for node_type in batch.node_types:
        assert (batch[node_type].num_sampled_nodes ==
                expected_num_sampled_nodes[node_type])
    for edge_type in batch.edge_types:
        assert (batch[edge_type].num_sampled_edges ==
                expected_num_sampled_edges[edge_type])


@withPackage('pyg_lib')
def test_neighbor_loader_mapping():
    edge_index = torch.tensor([
        [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 5],
        [1, 2, 3, 4, 5, 8, 6, 7, 9, 10, 6, 11],
    ])
    data = Data(edge_index=edge_index, num_nodes=12)

    loader = NeighborLoader(
        data,
        num_neighbors=[1],
        batch_size=2,
        shuffle=True,
    )

    for batch in loader:
        assert torch.equal(
            batch.n_id[batch.edge_index],
            data.edge_index[:, batch.e_id],
        )


@pytest.mark.skipif(
    not WITH_WEIGHTED_NEIGHBOR_SAMPLE,
    reason="'pyg-lib' does not support weighted neighbor sampling",
)
def test_weighted_homo_neighbor_loader():
    edge_index = torch.tensor([
        [1, 3, 0, 4],
        [2, 2, 1, 3],
    ])
    edge_weight = torch.tensor([0.0, 1.0, 0.0, 1.0])

    data = Data(num_nodes=5, edge_index=edge_index, edge_weight=edge_weight)

    loader = NeighborLoader(
        data,
        input_nodes=torch.tensor([2]),
        num_neighbors=[1] * 2,
        batch_size=1,
        weight_attr='edge_weight',
    )
    assert len(loader) == 1

    batch = next(iter(loader))

    assert batch.num_nodes == 3
    assert batch.n_id.tolist() == [2, 3, 4]
    assert batch.num_edges == 2
    assert batch.n_id[batch.edge_index].tolist() == [[3, 4], [2, 3]]


@pytest.mark.skipif(
    not WITH_WEIGHTED_NEIGHBOR_SAMPLE,
    reason="'pyg-lib' does not support weighted neighbor sampling",
)
def test_weighted_hetero_neighbor_loader():
    edge_index = torch.tensor([
        [1, 3, 0, 4],
        [2, 2, 1, 3],
    ])
    edge_weight = torch.tensor([0.0, 1.0, 0.0, 1.0])

    data = HeteroData()
    data['paper'].num_nodes = 5
    data['paper', 'to', 'paper'].edge_index = edge_index
    data['paper', 'to', 'paper'].edge_weight = edge_weight

    loader = NeighborLoader(
        data,
        input_nodes=('paper', torch.tensor([2])),
        num_neighbors=[1] * 2,
        batch_size=1,
        weight_attr='edge_weight',
    )
    assert len(loader) == 1

    batch = next(iter(loader))

    assert batch['paper'].num_nodes == 3
    assert batch['paper'].n_id.tolist() == [2, 3, 4]
    assert batch['paper', 'paper'].num_edges == 2
    global_edge_index = batch['paper'].n_id[batch['paper', 'paper'].edge_index]
    assert global_edge_index.tolist() == [[3, 4], [2, 3]]


@pytest.mark.skipif(
    not WITH_EDGE_TIME_NEIGHBOR_SAMPLE,
    reason="'pyg-lib' does not support weighted neighbor sampling",
)
def test_edge_level_temporal_homo_neighbor_loader():
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3],
    ])
    edge_time = torch.arange(edge_index.size(1))

    data = Data(edge_index=edge_index, edge_time=edge_time, num_nodes=5)

    loader = NeighborLoader(
        data,
        num_neighbors=[-1, -1],
        input_time=torch.tensor([4, 4, 4, 4, 4]),
        time_attr='edge_time',
        batch_size=1,
    )

    for batch in loader:
        assert batch.edge_time.numel() == batch.num_edges
        if batch.edge_time.numel() > 0:
            assert batch.edge_time.max() <= 4


@pytest.mark.skipif(
    not WITH_EDGE_TIME_NEIGHBOR_SAMPLE,
    reason="'pyg-lib' does not support weighted neighbor sampling",
)
def test_edge_level_temporal_hetero_neighbor_loader():
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3],
    ])
    edge_time = torch.arange(edge_index.size(1))

    data = HeteroData()
    data['A'].num_nodes = 5
    data['A', 'A'].edge_index = edge_index
    data['A', 'A'].edge_time = edge_time

    loader = NeighborLoader(
        data,
        num_neighbors=[-1, -1],
        input_nodes='A',
        input_time=torch.tensor([4, 4, 4, 4, 4]),
        time_attr='edge_time',
        batch_size=1,
    )

    for batch in loader:
        assert batch['A', 'A'].edge_time.numel() == batch['A', 'A'].num_edges
        if batch['A', 'A'].edge_time.numel() > 0:
            assert batch['A', 'A'].edge_time.max() <= 4


@withCUDA
@onlyNeighborSampler
@withPackage('torch_frame')
def test_neighbor_loader_with_tensor_frame(device):
    data = Data()
    data.tf = get_random_tensor_frame(num_rows=100, device=device)
    data.edge_index = get_random_edge_index(100, 100, 500, device=device)
    data.edge_attr = get_random_tensor_frame(500, device=device)
    data.global_tf = get_random_tensor_frame(num_rows=1, device=device)

    loader = NeighborLoader(data, num_neighbors=[5] * 2, batch_size=20)
    assert len(loader) == 5

    for batch in loader:
        assert isinstance(batch.tf, TensorFrame)
        assert batch.tf.device == device
        assert batch.tf.num_rows == batch.n_id.numel()
        assert batch.tf == data.tf[batch.n_id]

        assert isinstance(batch.edge_attr, TensorFrame)
        assert batch.edge_attr.device == device
        assert batch.edge_attr.num_rows == batch.e_id.numel()
        assert batch.edge_attr == data.edge_attr[batch.e_id]

        assert isinstance(batch.global_tf, TensorFrame)
        assert batch.global_tf.device == device
        assert batch.global_tf.num_rows == 1
        assert batch.global_tf == data.global_tf


@onlyNeighborSampler
def test_neighbor_loader_input_id():
    data = HeteroData()
    data['a'].num_nodes = 10
    data['b'].num_nodes = 12

    row = torch.randint(0, data['a'].num_nodes, (40, ))
    col = torch.randint(0, data['b'].num_nodes, (40, ))
    data['a', 'b'].edge_index = torch.stack([row, col], dim=0)
    data['b', 'a'].edge_index = torch.stack([col, row], dim=0)

    mask = torch.ones(data['a'].num_nodes, dtype=torch.bool)
    mask[0] = False

    loader = NeighborLoader(
        data,
        input_nodes=('a', mask),
        batch_size=2,
        num_neighbors=[2, 2],
    )
    for i, batch in enumerate(loader):
        if i < 4:
            expected = [(2 * i) + 1, (2 * i) + 2]
        else:
            expected = [(2 * i) + 1]

        assert batch['a'].input_id.tolist() == expected


@withPackage('pyg_lib')
def test_temporal_neighbor_loader_single_link():
    data = HeteroData()
    data['a'].x = torch.arange(10)
    data['b'].x = torch.arange(10)
    data['c'].x = torch.arange(10)

    data['b'].time = torch.arange(0, 10)
    data['c'].time = torch.arange(1, 11)

    data['a', 'b'].edge_index = torch.arange(10).view(1, -1).repeat(2, 1)
    data['b', 'a'].edge_index = torch.arange(10).view(1, -1).repeat(2, 1)
    data['a', 'c'].edge_index = torch.arange(10).view(1, -1).repeat(2, 1)
    data['c', 'a'].edge_index = torch.arange(10).view(1, -1).repeat(2, 1)

    loader = NeighborLoader(
        data,
        num_neighbors=[-1],
        input_nodes='a',
        time_attr='time',
        input_time=torch.arange(0, 10),
        batch_size=10,
    )
    batch = next(iter(loader))
    assert batch['a'].num_nodes == 10
    assert batch['b'].num_nodes == 10
    assert batch['c'].num_nodes == 0
