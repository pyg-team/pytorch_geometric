import itertools
import sys

import numpy as np
import pytest
import torch
from torch_sparse import SparseTensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.loader.neighbor_loader import get_input_nodes
from torch_geometric.nn import GraphConv, to_hetero
from torch_geometric.testing import withRegisteredOp
from torch_geometric.utils import k_hop_subgraph, sort_edge_index

sys.path.append("..")
# pylint: disable=wrong-import-order,wrong-import-position,no-name-in-module
from data.test_feature_store import MyFeatureStore  # noqa: E402
from data.test_graph_store import MyGraphStore  # noqa: E402


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.long)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.long)
    return torch.stack([row, col], dim=0)


def is_subset(subedge_index, edge_index, src_idx, dst_idx):
    num_nodes = int(edge_index.max()) + 1
    idx = num_nodes * edge_index[0] + edge_index[1]
    subidx = num_nodes * src_idx[subedge_index[0]] + dst_idx[subedge_index[1]]
    mask = torch.from_numpy(np.isin(subidx, idx))
    return int(mask.sum()) == mask.numel()


@pytest.mark.parametrize('directed', [True, False])
def test_homogeneous_neighbor_loader(directed):
    torch.manual_seed(12345)

    data = Data()

    data.x = torch.arange(100)
    data.edge_index = get_edge_index(100, 100, 500)
    data.edge_attr = torch.arange(500)

    loader = NeighborLoader(data, num_neighbors=[5] * 2, batch_size=20,
                            directed=directed)

    assert str(loader) == 'NeighborLoader()'
    assert len(loader) == 5

    for batch in loader:
        assert isinstance(batch, Data)

        assert len(batch) == 4
        assert batch.x.size(0) <= 100
        assert batch.batch_size == 20
        assert batch.x.min() >= 0 and batch.x.max() < 100
        assert batch.edge_index.min() >= 0
        assert batch.edge_index.max() < batch.num_nodes
        assert batch.edge_attr.min() >= 0
        assert batch.edge_attr.max() < 500

        assert is_subset(batch.edge_index, data.edge_index, batch.x, batch.x)


@pytest.mark.parametrize('directed', [True, False])
def test_heterogeneous_neighbor_loader(directed):
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

    r1, c1 = data['paper', 'paper'].edge_index
    r2, c2 = data['paper', 'author'].edge_index + torch.tensor([[0], [100]])
    r3, c3 = data['author', 'paper'].edge_index + torch.tensor([[100], [0]])
    full_adj = SparseTensor(
        row=torch.cat([r1, r2, r3]),
        col=torch.cat([c1, c2, c3]),
        value=torch.arange(2500),
    )

    batch_size = 20
    loader = NeighborLoader(data, num_neighbors=[10] * 2, input_nodes='paper',
                            batch_size=batch_size, directed=directed)
    assert str(loader) == 'NeighborLoader()'
    assert len(loader) == (100 + batch_size - 1) // batch_size

    for batch in loader:
        assert isinstance(batch, HeteroData)

        # Test node type selection:
        assert set(batch.node_types) == {'paper', 'author'}

        assert len(batch['paper']) == 2
        assert batch['paper'].x.size(0) <= 100
        assert batch['paper'].batch_size == batch_size
        assert batch['paper'].x.min() >= 0 and batch['paper'].x.max() < 100

        assert len(batch['author']) == 1
        assert batch['author'].x.size(0) <= 200
        assert batch['author'].x.min() >= 100 and batch['author'].x.max() < 300

        # Test edge type selection:
        assert set(batch.edge_types) == {('paper', 'to', 'paper'),
                                         ('paper', 'to', 'author'),
                                         ('author', 'to', 'paper')}

        assert len(batch['paper', 'paper']) == 2
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

        assert is_subset(batch['paper', 'paper'].edge_index,
                         data['paper', 'paper'].edge_index, batch['paper'].x,
                         batch['paper'].x)

        assert len(batch['paper', 'author']) == 2
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

        assert is_subset(batch['paper', 'author'].edge_index,
                         data['paper', 'author'].edge_index, batch['paper'].x,
                         batch['author'].x - 100)

        assert len(batch['author', 'paper']) == 2
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

        assert is_subset(batch['author', 'paper'].edge_index,
                         data['author', 'paper'].edge_index,
                         batch['author'].x - 100, batch['paper'].x)

        # Test for isolated nodes (there shouldn't exist any):
        n_id = torch.cat([batch['paper'].x, batch['author'].x])
        row, col, _ = full_adj[n_id, n_id].coo()
        assert torch.cat([row, col]).unique().numel() == n_id.numel()


@pytest.mark.parametrize('directed', [True, False])
def test_homogeneous_neighbor_loader_on_cora(get_dataset, directed):
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


@pytest.mark.parametrize('directed', [True, False])
def test_heterogeneous_neighbor_loader_on_cora(get_dataset, directed):
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


@withRegisteredOp('torch_sparse.hetero_temporal_neighbor_sample')
def test_temporal_heterogeneous_neighbor_loader_on_cora(get_dataset):
    dataset = get_dataset(name='Cora')
    data = dataset[0]

    hetero_data = HeteroData()
    hetero_data['paper'].x = data.x
    hetero_data['paper'].time = torch.arange(data.num_nodes)
    hetero_data['paper', 'paper'].edge_index = data.edge_index

    loader = NeighborLoader(hetero_data, num_neighbors=[-1, -1],
                            input_nodes='paper', time_attr='time',
                            batch_size=1)

    for batch in loader:
        mask = batch['paper'].time[0] >= batch['paper'].time[1:]
        assert torch.all(mask)


@pytest.mark.parametrize('directed', [True, False])
def test_custom_neighbor_loader(directed):
    r"""This test evaluates the correctness of a `NeighborLoader` constructed
    from a feature store and graph store by comparing it to a `NeighborLoader`
    constructed from a `HeteroData` object."""
    torch.manual_seed(12345)

    # Possible feature and graph stores:
    feature_stores = [MyFeatureStore, HeteroData]
    graph_stores = [MyGraphStore, HeteroData]

    # Set up edge indices:
    def _get_edge_index(num_src, num_dst, num_edges):
        edge_index = get_edge_index(num_src, num_dst, num_edges)
        edge_index = sort_edge_index(edge_index)
        adj = SparseTensor.from_edge_index(edge_index, is_sorted=True)
        rowptr, col, _ = adj.csr()
        return edge_index, rowptr, col

    # Assertion utility:
    def _assert_tensor_dict_equal(expected, actual):
        assert expected.keys() == actual.keys()
        for key in expected:
            assert torch.equal(expected[key], actual[key])

    # NOTE in this test, here we solely use explicit APIs, since
    # `HeteroData` and `Data` both override dunder methods:
    for feature_store, graph_store in itertools.product(
            feature_stores, graph_stores):

        # Initialize feature store, graph store, and reference:
        feature_store = feature_store()
        graph_store = graph_store()
        hetero_data = HeteroData()

        # Set up node features:
        x = torch.arange(100)
        hetero_data['paper'].x = x
        feature_store.put_tensor(x, group_name='paper', attr_name='x',
                                 index=None)
        x = torch.arange(100, 300)
        hetero_data['author'].x = x
        feature_store.put_tensor(x, group_name='author', attr_name='x',
                                 index=None)

        # Set up edge indices:
        edge_index, rowptr, col = _get_edge_index(100, 100, 500)
        hetero_data['paper', 'to', 'paper'].edge_index = edge_index
        graph_store.put_edge_index(
            edge_index=(rowptr, col),
            edge_type=('paper', 'to', 'paper'),
            layout='csr',
        )

        edge_index, rowptr, col = _get_edge_index(100, 200, 1000)
        hetero_data['paper', 'to', 'author'].edge_index = edge_index
        graph_store.put_edge_index(
            edge_index=(rowptr, col),
            edge_type=('paper', 'to', 'author'),
            layout='csr',
        )

        edge_index, rowptr, col = _get_edge_index(200, 100, 1000)
        hetero_data['author', 'to', 'paper'].edge_index = edge_index
        graph_store.put_edge_index(
            edge_index=(rowptr, col),
            edge_type=('author', 'to', 'paper'),
            layout='csr',
        )

        # Construct neighbor loaders:
        batch_size = 20
        input_type = 'paper'
        hetero_data_loader = NeighborLoader(
            data=hetero_data,
            num_neighbors=[-1] * 2,
            input_nodes=input_type,
            batch_size=batch_size,
            directed=directed,
        )

        input_type = feature_store._tensor_attr_cls(group_name='paper',
                                                    attr_name='x')
        custom_loader = NeighborLoader(
            data=(feature_store, graph_store),
            input_nodes=input_type,
            num_neighbors=[-1] * 2,
            batch_size=batch_size,
            is_sorted=True,
            directed=directed,
        )

        # Basic assertions:
        assert str(custom_loader) == 'NeighborLoader()'
        assert len(custom_loader) == (100 + batch_size - 1) // batch_size

        # Equivalent input nodes:
        hetero_input_nodes = get_input_nodes(hetero_data,
                                             hetero_data_loader.input_nodes)
        custom_input_nodes = get_input_nodes((feature_store, graph_store),
                                             custom_loader.input_nodes)

        assert hetero_input_nodes == custom_input_nodes

        # Equivalent inner representations:
        assert (hetero_data_loader.neighbor_sampler.node_types ==
                custom_loader.neighbor_sampler.node_types)
        assert (hetero_data_loader.neighbor_sampler.edge_types ==
                custom_loader.neighbor_sampler.edge_types)

        # Equivalent neighbor sampler outputs:
        expected = hetero_data_loader.neighbor_sampler([0, 1, 2, 3])
        actual = custom_loader.neighbor_sampler([0, 1, 2, 3])

        for i in range(len(expected) - 1):
            _assert_tensor_dict_equal(expected[i], actual[i])

        # Equivalent outputs when iterating the `DataLoader`:
        custom_batches = []
        for batch in custom_loader:
            assert isinstance(batch, HeteroData)
            custom_batches.append(batch)

        hetero_data_batches = []
        for batch in hetero_data_loader:
            hetero_data_batches.append(batch)

        for expected, actual in zip(hetero_data_batches, custom_batches):
            # Check node features:
            for node_type in actual.node_types:
                assert torch.equal(expected[node_type].x, actual[node_type].x)

            # Check edge indices:
            for edge_type in actual.edge_types:
                assert torch.equal(expected[edge_type].edge_index,
                                   actual[edge_type].edge_index)
