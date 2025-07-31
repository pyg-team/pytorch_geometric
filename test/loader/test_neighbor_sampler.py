import numpy as np
import torch

from torch_geometric.loader import NeighborSampler
from torch_geometric.nn.conv import GATConv, SAGEConv
from torch_geometric.testing import onlyOnline, withPackage
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import erdos_renyi_graph


@withPackage('torch_sparse')
def test_neighbor_sampler_basic():
    edge_index = erdos_renyi_graph(num_nodes=10, edge_prob=0.5)
    adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(10, 10)).t()
    E = edge_index.size(1)

    loader = NeighborSampler(edge_index, sizes=[2, 4], batch_size=2)
    assert str(loader) == 'NeighborSampler(sizes=[2, 4])'
    assert len(loader) == 5

    for batch_size, n_id, adjs in loader:
        assert batch_size == 2
        assert all(np.isin(n_id, torch.arange(10)).tolist())
        assert n_id.unique().size(0) == n_id.size(0)
        for (edge_index, e_id, size) in adjs:
            assert int(edge_index[0].max() + 1) <= size[0]
            assert int(edge_index[1].max() + 1) <= size[1]
            assert all(np.isin(e_id, torch.arange(E)).tolist())
            assert e_id.unique().size(0) == e_id.size(0)
            assert size[0] >= size[1]

    out = loader.sample([1, 2])
    assert len(out) == 3

    loader = NeighborSampler(adj_t, sizes=[2, 4], batch_size=2)

    for _, _, adjs in loader:
        for adj_t, _, size in adjs:
            assert adj_t.size(0) == size[1]
            assert adj_t.size(1) == size[0]


@withPackage('torch_sparse')
def test_neighbor_sampler_invalid_kwargs():
    # Ignore `collate_fn` and `dataset` arguments:
    edge_index = torch.tensor([[0, 1], [1, 0]])
    NeighborSampler(edge_index, sizes=[-1], collate_fn=None, dataset=None)


@onlyOnline
@withPackage('torch_sparse')
def test_neighbor_sampler_on_cora(get_dataset):
    dataset = get_dataset(name='Cora')
    data = dataset[0]

    batch = torch.arange(10)
    loader = NeighborSampler(data.edge_index, sizes=[-1, -1, -1],
                             node_idx=batch, batch_size=10)

    class SAGE(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()

            self.convs = torch.nn.ModuleList()
            self.convs.append(SAGEConv(in_channels, 16))
            self.convs.append(SAGEConv(16, 16))
            self.convs.append(SAGEConv(16, out_channels))

        def batch(self, x, adjs):
            for i, (edge_index, _, size) in enumerate(adjs):
                x_target = x[:size[1]]  # Target nodes are always placed first.
                x = self.convs[i]((x, x_target), edge_index)
            return x

        def full(self, x, edge_index):
            for conv in self.convs:
                x = conv(x, edge_index)
            return x

    model = SAGE(dataset.num_features, dataset.num_classes)

    _, n_id, adjs = next(iter(loader))
    out1 = model.batch(data.x[n_id], adjs)
    out2 = model.full(data.x, data.edge_index)[batch]
    assert torch.allclose(out1, out2, atol=1e-7)

    class GAT(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()

            self.convs = torch.nn.ModuleList()
            self.convs.append(GATConv(in_channels, 16, heads=2))
            self.convs.append(GATConv(32, 16, heads=2))
            self.convs.append(GATConv(32, out_channels, heads=2, concat=False))

        def batch(self, x, adjs):
            for i, (edge_index, _, size) in enumerate(adjs):
                x_target = x[:size[1]]  # Target nodes are always placed first.
                x = self.convs[i]((x, x_target), edge_index)
            return x

        def full(self, x, edge_index):
            for conv in self.convs:
                x = conv(x, edge_index)
            return x

    _, n_id, adjs = next(iter(loader))
    out1 = model.batch(data.x[n_id], adjs)
    out2 = model.full(data.x, data.edge_index)[batch]
    assert torch.allclose(out1, out2, atol=1e-7)
