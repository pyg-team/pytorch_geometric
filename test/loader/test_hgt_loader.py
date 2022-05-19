import numpy as np
import torch
from torch_sparse import SparseTensor

from torch_geometric.data import HeteroData
from torch_geometric.loader import HGTLoader
from torch_geometric.nn import GraphConv, to_hetero
from torch_geometric.utils import k_hop_subgraph


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


def test_hgt_loader():
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
    loader = HGTLoader(data, num_samples=[5] * 4, batch_size=batch_size,
                       input_nodes='paper')
    assert str(loader) == 'HGTLoader()'
    assert len(loader) == (100 + batch_size - 1) // batch_size

    for batch in loader:
        assert isinstance(batch, HeteroData)

        # Test node type selection:
        assert set(batch.node_types) == {'paper', 'author'}

        assert len(batch['paper']) == 2
        assert batch['paper'].x.size() == (40, )  # 20 + 4 * 5
        assert batch['paper'].batch_size == batch_size
        assert batch['paper'].x.min() >= 0 and batch['paper'].x.max() < 100

        assert len(batch['author']) == 1
        assert batch['author'].x.size() == (20, )  # 4 * 5
        assert batch['author'].x.min() >= 100 and batch['author'].x.max() < 300

        # Test edge type selection:
        assert set(batch.edge_types) == {('paper', 'to', 'paper'),
                                         ('paper', 'to', 'author'),
                                         ('author', 'to', 'paper')}

        assert len(batch['paper', 'paper']) == 2
        row, col = batch['paper', 'paper'].edge_index
        value = batch['paper', 'paper'].edge_attr
        adj = full_adj[batch['paper'].x, batch['paper'].x]
        assert row.min() >= 0 and row.max() < 40
        assert col.min() >= 0 and col.max() < 40
        assert value.min() >= 0 and value.max() < 500
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
        adj = full_adj[batch['paper'].x, batch['author'].x]
        assert row.min() >= 0 and row.max() < 40
        assert col.min() >= 0 and col.max() < 20
        assert value.min() >= 500 and value.max() < 1500
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
        adj = full_adj[batch['author'].x, batch['paper'].x]
        assert row.min() >= 0 and row.max() < 20
        assert col.min() >= 0 and col.max() < 40
        assert value.min() >= 1500 and value.max() < 2500
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
        assert torch.cat([row, col]).unique().numel() >= 59


def test_hgt_loader_on_cora(get_dataset):
    dataset = get_dataset(name='Cora')
    data = dataset[0]
    data.edge_weight = torch.rand(data.num_edges)

    hetero_data = HeteroData()
    hetero_data['paper'].x = data.x
    hetero_data['paper'].n_id = torch.arange(data.num_nodes)
    hetero_data['paper', 'paper'].edge_index = data.edge_index
    hetero_data['paper', 'paper'].edge_weight = data.edge_weight

    split_idx = torch.arange(5, 8)

    # Sample the complete two-hop neighborhood:
    loader = HGTLoader(hetero_data, num_samples=[data.num_nodes] * 2,
                       batch_size=split_idx.numel(),
                       input_nodes=('paper', split_idx))
    assert len(loader) == 1

    hetero_batch = next(iter(loader))
    batch_size = hetero_batch['paper'].batch_size

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
