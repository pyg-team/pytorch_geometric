import torch
from torch_sparse import SparseTensor

from torch_geometric.data import Data
from torch_geometric.transforms import GCNNorm


def test_gcn_norm():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_weight = torch.ones(edge_index.size(1))
    adj_t = SparseTensor.from_edge_index(edge_index, edge_weight).t()

    transform = GCNNorm()
    assert str(transform) == 'GCNNorm(add_self_loops=True)'

    expected_edge_index = [[0, 1, 1, 2, 0, 1, 2], [1, 0, 2, 1, 0, 1, 2]]
    expected_edge_weight = torch.tensor(
        [0.4082, 0.4082, 0.4082, 0.4082, 0.5000, 0.3333, 0.5000])

    data = Data(edge_index=edge_index, edge_weight=edge_weight, num_nodes=3)
    data = transform(data)
    assert len(data) == 3
    assert data.num_nodes == 3
    assert data.edge_index.tolist() == expected_edge_index
    assert torch.allclose(data.edge_weight, expected_edge_weight, atol=1e-4)

    data = Data(edge_index=edge_index, num_nodes=3)
    data = transform(data)
    assert len(data) == 3
    assert data.num_nodes == 3
    assert data.edge_index.tolist() == expected_edge_index
    assert torch.allclose(data.edge_weight, expected_edge_weight, atol=1e-4)

    # For `SparseTensor`, expected outputs will be sorted:
    expected_edge_index = [[0, 0, 1, 1, 1, 2, 2], [0, 1, 0, 1, 2, 1, 2]]
    expected_edge_weight = torch.tensor(
        [0.500, 0.4082, 0.4082, 0.3333, 0.4082, 0.4082, 0.5000])

    data = Data(adj_t=adj_t)
    data = transform(data)
    assert len(data) == 1
    row, col, value = data.adj_t.coo()
    assert row.tolist() == expected_edge_index[0]
    assert col.tolist() == expected_edge_index[1]
    assert torch.allclose(value, expected_edge_weight, atol=1e-4)
