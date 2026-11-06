import pytest
import torch

import torch_geometric.typing
from torch_geometric.data import Data
from torch_geometric.transforms import GCNNorm, ToSparseTensor
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_edge_index


def test_gcn_norm():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_weight = torch.ones(edge_index.size(1))

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
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        expected_edge_index = [[0, 0, 1, 1, 1, 2, 2], [0, 1, 0, 1, 2, 1, 2]]
        expected_edge_weight = torch.tensor(
            [0.500, 0.4082, 0.4082, 0.3333, 0.4082, 0.4082, 0.5000])

        adj_t = SparseTensor.from_edge_index(edge_index, edge_weight).t()
        data = Data(adj_t=adj_t)
        data = transform(data)
        assert len(data) == 1
        row, col, value = data.adj_t.coo()
        assert row.tolist() == expected_edge_index[0]
        assert col.tolist() == expected_edge_index[1]
        assert torch.allclose(value, expected_edge_weight, atol=1e-4)


@pytest.mark.parametrize('layout', [None, torch.sparse_coo, torch.sparse_csr])
def test_gcn_norm_adj_t(layout):
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    data = Data(edge_index=edge_index, num_nodes=3)
    data = ToSparseTensor(layout=layout)(data)
    data = GCNNorm()(data)

    if layout is None and torch_geometric.typing.WITH_TORCH_SPARSE:
        assert isinstance(data.adj_t, SparseTensor)
    else:
        assert data.adj_t.layout == (layout or torch.sparse_csr)

    expected_edge_index = [[0, 0, 1, 1, 1, 2, 2], [0, 1, 0, 1, 2, 1, 2]]
    expected_edge_weight = torch.tensor(
        [0.500, 0.4082, 0.4082, 0.3333, 0.4082, 0.4082, 0.5000])

    edge_index, edge_weight = to_edge_index(data.adj_t)
    assert edge_index.tolist() == expected_edge_index
    assert torch.allclose(edge_weight, expected_edge_weight, atol=1e-4)
