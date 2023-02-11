import pytest
import torch
from torch_sparse import SparseTensor

from torch_geometric.data import Data
from torch_geometric.transforms import GCNNorm


@pytest.mark.parametrize('add_self_loops', [True, False])
def test_gcn_norm(add_self_loops):
    assert GCNNorm().__repr__() == 'GCNNorm()'

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_attr = torch.Tensor([1, 1, 1, 1])
    if add_self_loops:
        expected_edge_index = [[0, 1, 1, 2, 0, 1, 2], [1, 0, 2, 1, 0, 1, 2]]
        expected_edge_weight = torch.tensor(
            [0.4082, 0.4082, 0.4082, 0.4082, 0.5000, 0.3333, 0.5000])
    else:
        expected_edge_index = edge_index.tolist()
        expected_edge_weight = torch.tensor(
            [0.70711, 0.70711, 0.70711, 0.70711])

    # Test with edge_index as LongTensor:
    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=3)
    data = GCNNorm(add_self_loops=add_self_loops)(data)
    assert data.edge_index.tolist() == expected_edge_index
    assert torch.allclose(data.edge_weight, expected_edge_weight)

    data = Data(edge_index=edge_index, edge_weight=edge_attr, num_nodes=3)
    data = GCNNorm(add_self_loops=add_self_loops)(data)
    assert data.edge_index.tolist() == expected_edge_index
    assert torch.allclose(data.edge_weight, expected_edge_weight)

    # Test with edge_index as SparseTensor:
    adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(3, 3))
    data = Data(adj_t=adj_t)
    data = GCNNorm(add_self_loops=add_self_loops)(data)
    assert data.adj_t.storage.row().tolist() == expected_edge_index[0]
    assert data.adj_t.storage.col().tolist() == expected_edge_index[1]
    assert torch.allclose(data.adj_t.storage.value(), expected_edge_weight)
