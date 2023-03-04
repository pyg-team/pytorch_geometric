import os.path as osp
import subprocess
from time import sleep

import numpy as np
import pytest
import torch
from torch_sparse import SparseTensor

import torch_geometric.typing
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GraphConv, to_hetero
from torch_geometric.testing import (
    MyFeatureStore,
    MyGraphStore,
    onlyLinux,
    withPackage,
)
from torch_geometric.typing import WITH_PYG_LIB
from torch_geometric.utils import k_hop_subgraph


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges, dtype=torch.int64):
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=dtype)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=dtype)
    return torch.stack([row, col], dim=0)


def is_subset(subedge_index, edge_index, src_idx, dst_idx):
    num_nodes = int(edge_index.max()) + 1
    idx = num_nodes * edge_index[0] + edge_index[1]
    subidx = num_nodes * src_idx[subedge_index[0]] + dst_idx[subedge_index[1]]
    mask = torch.from_numpy(np.isin(subidx, idx))
    return int(mask.sum()) == mask.numel()


@pytest.mark.parametrize('directed', [True])  # TODO re-enable undirected mode
@pytest.mark.parametrize('dtype', [torch.int64, torch.int32])
def test_homo_neighbor_loader_basic(directed, dtype):
    if dtype != torch.int64 and not torch_geometric.typing.WITH_PYG_LIB:
        return

    torch.manual_seed(12345)

    data = Data()

    data.x = torch.arange(100)
    data.edge_index = get_edge_index(100, 100, 500, dtype)
    data.edge_attr = torch.arange(500)

    loader = NeighborLoader(data, num_neighbors=[5] * 2, batch_size=20,
                            directed=directed)

    assert str(loader) == 'NeighborLoader()'
    assert len(loader) == 5

    for i, batch in enumerate(loader):
        assert isinstance(batch, Data)
        assert len(batch) == 9 if WITH_PYG_LIB else 7
        assert batch.x.size(0) <= 100
        assert batch.n_id.size() == (batch.num_nodes, )
        assert batch.e_id.size() == (batch.num_edges, )
        assert batch.input_id.numel() == batch.batch_size == 20
        assert batch.x.min() >= 0 and batch.x.max() < 100
        assert batch.edge_index.min() >= 0
        assert batch.edge_index.max() < batch.num_nodes
        assert batch.edge_attr.min() >= 0
        assert batch.edge_attr.max() < 500

        # Input nodes are always sampled first:
        assert torch.equal(
            batch.x[:batch.batch_size],
            torch.arange(i * batch.batch_size, (i + 1) * batch.batch_size))

        assert is_subset(
            batch.edge_index.to(torch.int64),
            data.edge_index.to(torch.int64),
            batch.x,
            batch.x,
        )


@pytest.mark.parametrize('directed', [True])  # TODO re-enable undirected mode
@pytest.mark.parametrize('dtype', [torch.int64, torch.int32])
def test_hetero_neighbor_loader_basic(directed, dtype):
    if dtype != torch.int64 and not torch_geometric.typing.WITH_PYG_LIB:
        return

    torch.manual_seed(12345)

    data = HeteroData()

    data['paper'].x = torch.arange(100)
    data['author'].x = torch.arange(100, 300)

    data['paper', 'paper'].edge_index = get_edge_index(100, 100, 500, dtype)
    data['paper', 'paper'].edge_attr = torch.arange(500)
    data['paper', 'author'].edge_index = get_edge_index(100, 200, 1000, dtype)
    data['paper', 'author'].edge_attr = torch.arange(500, 1500)
    data['author', 'paper'].edge_index = get_edge_index(200, 100, 1000, dtype)
    data['author', 'paper'].edge_attr = torch.arange(1500, 2500)

    r1, c1 = data['paper', 'paper'].edge_index
    r2, c2 = data['paper', 'author'].edge_index + torch.tensor([[0], [100]])
    r3, c3 = data['author', 'paper'].edge_index + torch.tensor([[100], [0]])
    full_adj = SparseTensor(
        row=torch.cat([r1, r2, r3]),
        col=torch.cat([c1, c2, c3]),
        value=torch.arange(2500),
    )

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
            directed=directed,
        )
        next(iter(loader))

    loader = NeighborLoader(
        data,
        num_neighbors=[10] * 2,
        input_nodes='paper',
        batch_size=batch_size,
        directed=directed,
    )

    assert str(loader) == 'NeighborLoader()'
    assert len(loader) == (100 + batch_size - 1) // batch_size

    for batch in loader:
        assert isinstance(batch, HeteroData)

        # Test node type selection:
        assert set(batch.node_types) == {'paper', 'author'}

        assert len(batch['paper']) == 5 if WITH_PYG_LIB else 4
        assert batch['paper'].n_id.size() == (batch['paper'].num_nodes, )
        assert batch['paper'].x.size(0) <= 100
        assert batch['paper'].input_id.numel() == batch_size
        assert batch['paper'].batch_size == batch_size
        assert batch['paper'].x.min() >= 0 and batch['paper'].x.max() < 100

        assert len(batch['author']) == 3 if WITH_PYG_LIB else 2
        assert batch['author'].n_id.size() == (batch['author'].num_nodes, )
        assert batch['author'].x.size(0) <= 200
        assert batch['author'].x.min() >= 100 and batch['author'].x.max() < 300

        # Test edge type selection:
        assert set(batch.edge_types) == {('paper', 'to', 'paper'),
                                         ('paper', 'to', 'author'),
                                         ('author', 'to', 'paper')}

        assert len(batch['paper', 'paper']) == 4 if WITH_PYG_LIB else 3
        num_edges = batch['paper', 'paper'].num_edges
        assert batch['paper', 'paper'].e_id.size() == (num_edges, )
        row, col = batch['paper', 'paper'].edge_index
        value = batch['paper', 'paper'].edge_attr
        assert row.min() >= 0 and row.max() < batch['paper'].num_nodes
        assert col.min() >= 0 and col.max() < batch['paper'].num_nodes
        assert value.min() >= 0 and value.max() < 500
        if not directed:
            adj = full_adj[batch['paper'].x, batch['paper'].x]
            assert adj.nnz() == row.size(0)
            assert torch.allclose(row.unique(), adj.storage.row().unique())
            assert torch.allclose(col.unique(), adj.storage.col().unique())
            assert torch.allclose(value.unique(), adj.storage.value().unique())

        assert is_subset(
            batch['paper', 'paper'].edge_index.to(torch.int64),
            data['paper', 'paper'].edge_index.to(torch.int64),
            batch['paper'].x,
            batch['paper'].x,
        )

        assert len(batch['paper', 'author']) == 4 if WITH_PYG_LIB else 3
        num_edges = batch['paper', 'author'].num_edges
        assert batch['paper', 'author'].e_id.size() == (num_edges, )
        row, col = batch['paper', 'author'].edge_index
        value = batch['paper', 'author'].edge_attr
        assert row.min() >= 0 and row.max() < batch['paper'].num_nodes
        assert col.min() >= 0 and col.max() < batch['author'].num_nodes
        assert value.min() >= 500 and value.max() < 1500
        if not directed:
            adj = full_adj[batch['paper'].x, batch['author'].x]
            assert adj.nnz() == row.size(0)
            assert torch.allclose(row.unique(), adj.storage.row().unique())
            assert torch.allclose(col.unique(), adj.storage.col().unique())
            assert torch.allclose(value.unique(), adj.storage.value().unique())

        assert is_subset(
            batch['paper', 'author'].edge_index.to(torch.int64),
            data['paper', 'author'].edge_index.to(torch.int64),
            batch['paper'].x,
            batch['author'].x - 100,
        )

        assert len(batch['author', 'paper']) == 4 if WITH_PYG_LIB else 3
        num_edges = batch['author', 'paper'].num_edges
        assert batch['author', 'paper'].e_id.size() == (num_edges, )
        row, col = batch['author', 'paper'].edge_index
        value = batch['author', 'paper'].edge_attr
        assert row.min() >= 0 and row.max() < batch['author'].num_nodes
        assert col.min() >= 0 and col.max() < batch['paper'].num_nodes
        assert value.min() >= 1500 and value.max() < 2500
        if not directed:
            adj = full_adj[batch['author'].x, batch['paper'].x]
            assert adj.nnz() == row.size(0)
            assert torch.allclose(row.unique(), adj.storage.row().unique())
            assert torch.allclose(col.unique(), adj.storage.col().unique())
            assert torch.allclose(value.unique(), adj.storage.value().unique())

        assert is_subset(
            batch['author', 'paper'].edge_index.to(torch.int64),
            data['author', 'paper'].edge_index.to(torch.int64),
            batch['author'].x - 100,
            batch['paper'].x,
        )

        # Test for isolated nodes (there shouldn't exist any):
        n_id = torch.cat([batch['paper'].x, batch['author'].x])
        row, col, _ = full_adj[n_id, n_id].coo()
        assert torch.cat([row, col]).unique().numel() == n_id.numel()


@pytest.mark.parametrize('directed', [True])  # TODO re-enable undirected mode
def test_homo_neighbor_loader_on_cora(get_dataset, directed):
    dataset = get_dataset(name='Cora')
    data = dataset[0]
    data.n_id = torch.arange(data.num_nodes)
    data.edge_weight = torch.rand(data.num_edges)

    split_idx = torch.arange(5, 8)

    loader = NeighborLoader(data, num_neighbors=[-1, -1],
                            batch_size=split_idx.numel(),
                            input_nodes=split_idx, directed=directed)
    assert len(loader) == 1

    batch = next(iter(loader))
    batch_size = batch.batch_size

    if not directed:
        n_id, _, _, e_mask = k_hop_subgraph(split_idx, num_hops=2,
                                            edge_index=data.edge_index,
                                            num_nodes=data.num_nodes)

        assert n_id.sort()[0].tolist() == batch.n_id.sort()[0].tolist()
        assert batch.num_edges == int(e_mask.sum())

    class GNN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = GraphConv(in_channels, hidden_channels)
            self.conv2 = GraphConv(hidden_channels, out_channels)

        def forward(self, x, edge_index, edge_weight):
            x = self.conv1(x, edge_index, edge_weight).relu()
            x = self.conv2(x, edge_index, edge_weight).relu()
            return x

    model = GNN(dataset.num_features, 16, dataset.num_classes)

    out1 = model(data.x, data.edge_index, data.edge_weight)[split_idx]
    out2 = model(batch.x, batch.edge_index, batch.edge_weight)[:batch_size]
    assert torch.allclose(out1, out2, atol=1e-6)


@pytest.mark.parametrize('directed', [True])  # TODO re-enable undirected mode
def test_hetero_neighbor_loader_on_cora(get_dataset, directed):
    dataset = get_dataset(name='Cora')
    data = dataset[0]
    data.edge_weight = torch.rand(data.num_edges)

    hetero_data = HeteroData()
    hetero_data['paper'].x = data.x
    hetero_data['paper'].n_id = torch.arange(data.num_nodes)
    hetero_data['paper', 'paper'].edge_index = data.edge_index
    hetero_data['paper', 'paper'].edge_weight = data.edge_weight

    split_idx = torch.arange(5, 8)

    loader = NeighborLoader(hetero_data, num_neighbors=[-1, -1],
                            batch_size=split_idx.numel(),
                            input_nodes=('paper', split_idx),
                            directed=directed)
    assert len(loader) == 1

    hetero_batch = next(iter(loader))
    batch_size = hetero_batch['paper'].batch_size

    if not directed:
        n_id, _, _, e_mask = k_hop_subgraph(split_idx, num_hops=2,
                                            edge_index=data.edge_index,
                                            num_nodes=data.num_nodes)

        n_id = n_id.sort()[0]
        assert n_id.tolist() == hetero_batch['paper'].n_id.sort()[0].tolist()
        assert hetero_batch['paper', 'paper'].num_edges == int(e_mask.sum())

    class GNN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = GraphConv(in_channels, hidden_channels)
            self.conv2 = GraphConv(hidden_channels, out_channels)

        def forward(self, x, edge_index, edge_weight):
            x = self.conv1(x, edge_index, edge_weight).relu()
            x = self.conv2(x, edge_index, edge_weight).relu()
            return x

    model = GNN(dataset.num_features, 16, dataset.num_classes)
    hetero_model = to_hetero(model, hetero_data.metadata())

    out1 = model(data.x, data.edge_index, data.edge_weight)[split_idx]
    out2 = hetero_model(hetero_batch.x_dict, hetero_batch.edge_index_dict,
                        hetero_batch.edge_weight_dict)['paper'][:batch_size]
    assert torch.allclose(out1, out2, atol=1e-6)


@withPackage('pyg_lib')
def test_temporal_hetero_neighbor_loader_on_cora(get_dataset):
    dataset = get_dataset(name='Cora')
    data = dataset[0]

    hetero_data = HeteroData()
    hetero_data['paper'].x = data.x
    hetero_data['paper'].time = torch.arange(data.num_nodes, 0, -1)
    hetero_data['paper', 'paper'].edge_index = data.edge_index

    loader = NeighborLoader(hetero_data, num_neighbors=[-1, -1],
                            input_nodes='paper', time_attr='time',
                            batch_size=1)

    for batch in loader:
        mask = batch['paper'].time[0] >= batch['paper'].time[1:]
        assert torch.all(mask)


@pytest.mark.parametrize('FeatureStore', [MyFeatureStore, HeteroData])
@pytest.mark.parametrize('GraphStore', [MyGraphStore, HeteroData])
def test_custom_neighbor_loader(FeatureStore, GraphStore):
    # Initialize feature store, graph store, and reference:
    feature_store = FeatureStore()
    graph_store = GraphStore()
    data = HeteroData()

    # Set up node features:
    x = torch.arange(100)
    data['paper'].x = x
    feature_store.put_tensor(x, group_name='paper', attr_name='x', index=None)

    x = torch.arange(100, 300)
    data['author'].x = x
    feature_store.put_tensor(x, group_name='author', attr_name='x', index=None)

    # COO:
    edge_index = get_edge_index(100, 100, 500)
    data['paper', 'to', 'paper'].edge_index = edge_index
    coo = (edge_index[0], edge_index[1])
    graph_store.put_edge_index(edge_index=coo,
                               edge_type=('paper', 'to', 'paper'),
                               layout='coo', size=(100, 100))

    # CSR:
    edge_index = get_edge_index(100, 200, 1000)
    data['paper', 'to', 'author'].edge_index = edge_index
    csr = SparseTensor.from_edge_index(edge_index).csr()[:2]
    graph_store.put_edge_index(edge_index=csr,
                               edge_type=('paper', 'to', 'author'),
                               layout='csr', size=(100, 200))

    # CSC:
    edge_index = get_edge_index(200, 100, 1000)
    data['author', 'to', 'paper'].edge_index = edge_index
    csc = SparseTensor(row=edge_index[1], col=edge_index[0]).csr()[-2::-1]
    graph_store.put_edge_index(edge_index=csc,
                               edge_type=('author', 'to', 'paper'),
                               layout='csc', size=(200, 100))

    # COO (sorted):
    edge_index = get_edge_index(200, 200, 100)
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
        # loader2 explicitly adds `num_nodes` to the batch
        assert len(batch1) + 1 == len(batch2)
        assert batch1['paper'].batch_size == batch2['paper'].batch_size

        # Mapped indices of neighbors may be differently sorted:
        assert torch.allclose(batch1['paper'].x.sort()[0],
                              batch2['paper'].x.sort()[0])
        assert torch.allclose(batch1['author'].x.sort()[0],
                              batch2['author'].x.sort()[0])

        assert (batch1['paper', 'to', 'paper'].edge_index.size() == batch1[
            'paper', 'to', 'paper'].edge_index.size())
        assert (batch1['paper', 'to', 'author'].edge_index.size() == batch1[
            'paper', 'to', 'author'].edge_index.size())
        assert (batch1['author', 'to', 'paper'].edge_index.size() == batch1[
            'author', 'to', 'paper'].edge_index.size())


@withPackage('pyg_lib')
@pytest.mark.parametrize('FeatureStore', [MyFeatureStore, HeteroData])
@pytest.mark.parametrize('GraphStore', [MyGraphStore, HeteroData])
def test_temporal_custom_neighbor_loader_on_cora(get_dataset, FeatureStore,
                                                 GraphStore):
    # Initialize dataset (once):
    dataset = get_dataset(name='Cora')
    data = dataset[0]
    data.time = torch.arange(data.num_nodes, 0, -1)

    # Initialize feature store, graph store, and reference:
    feature_store = FeatureStore()
    graph_store = GraphStore()
    hetero_data = HeteroData()

    feature_store.put_tensor(
        data.x,
        group_name='paper',
        attr_name='x',
        index=None,
    )
    hetero_data['paper'].x = data.x

    feature_store.put_tensor(
        data.time,
        group_name='paper',
        attr_name='time',
        index=None,
    )
    hetero_data['paper'].time = data.time

    # Sort according to time in local neighborhoods:
    row, col = data.edge_index
    perm = ((col * (data.num_nodes + 1)) + data.time[row]).argsort()
    edge_index = data.edge_index[:, perm]

    graph_store.put_edge_index(
        edge_index,
        edge_type=('paper', 'to', 'paper'),
        layout='coo',
        is_sorted=True,
        size=(data.num_nodes, data.num_nodes),
    )
    hetero_data['paper', 'to', 'paper'].edge_index = data.edge_index

    loader1 = NeighborLoader(
        hetero_data,
        num_neighbors=[-1, -1],
        input_nodes='paper',
        time_attr='time',
        batch_size=128,
    )

    loader2 = NeighborLoader(
        (feature_store, graph_store),
        num_neighbors=[-1, -1],
        input_nodes='paper',
        time_attr='time',
        batch_size=128,
    )

    for batch1, batch2 in zip(loader1, loader2):
        assert torch.equal(batch1['paper'].time, batch2['paper'].time)


@withPackage('pyg_lib')
def test_pyg_lib_homo_neighbor_loader():
    adj = SparseTensor.from_edge_index(get_edge_index(20, 20, 100))
    colptr, row, _ = adj.csc()

    seed = torch.arange(10)

    sample = torch.ops.pyg.neighbor_sample
    out1 = sample(colptr, row, seed, [-1, -1], None, None, True)
    sample = torch.ops.torch_sparse.neighbor_sample
    out2 = sample(colptr, row, seed, [-1, -1], False, True)

    row1, col1, node_id1, edge_id1 = out1[:4]
    node_id2, row2, col2, edge_id2 = out2
    assert torch.equal(node_id1, node_id2)
    assert torch.equal(row1, row2)
    assert torch.equal(col1, col2)
    assert torch.equal(edge_id1, edge_id2)


@withPackage('pyg_lib')
def test_pyg_lib_hetero_neighbor_loader():
    adj1 = SparseTensor.from_edge_index(get_edge_index(20, 10, 50))
    colptr1, row1, _ = adj1.csc()

    adj2 = SparseTensor.from_edge_index(get_edge_index(10, 20, 50))
    colptr2, row2, _ = adj2.csc()

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
                  num_neighbors_dict, None, None, True, False, True, False,
                  "uniform", True)
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
def test_memmap_neighbor_loader(tmp_path):
    path = osp.join(tmp_path, 'x.npy')
    x = np.memmap(path, dtype=np.float32, mode='w+', shape=(100, 32))
    x[:] = np.random.randn(100, 32)

    data = Data()
    data.x = np.memmap(path, dtype=np.float32, mode='r', shape=(100, 32))
    data.edge_index = get_edge_index(100, 100, 500)

    assert str(data) == 'Data(x=[100, 32], edge_index=[2, 500])'
    assert data.num_nodes == 100

    loader = NeighborLoader(data, num_neighbors=[5] * 2, batch_size=20,
                            num_workers=2)
    batch = next(iter(loader))
    assert batch.num_nodes <= 100
    assert isinstance(batch.x, torch.Tensor)
    assert batch.x.size() == (batch.num_nodes, 32)


@onlyLinux
@pytest.mark.parametrize('num_workers,loader_cores', [
    (1, None),
    (1, [1]),
])
def test_cpu_affinity_neighbor_loader(num_workers, loader_cores):
    data = Data(x=torch.randn(1, 1))
    loader = NeighborLoader(data, num_neighbors=[-1], batch_size=1,
                            num_workers=num_workers)

    if isinstance(loader_cores, list):
        loader_cores = loader_cores[:num_workers]

    out = []
    with loader.enable_cpu_affinity(loader_cores):
        iterator = loader._get_iterator().iterator
        workers = iterator._workers
        for worker in workers:
            sleep(1)  # Gives time for worker to initialize.
            process = subprocess.Popen(
                ['taskset', '-c', '-p', f'{worker.pid}'],
                stdout=subprocess.PIPE)
            stdout = process.communicate()[0].decode('utf-8')
            out.append(int(stdout.split(':')[1].strip()))
        if not loader_cores:
            assert out == list(range(0, num_workers))
        else:
            assert out == loader_cores


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
