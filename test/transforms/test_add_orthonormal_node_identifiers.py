import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from torch_geometric.data import Data
from torch_geometric.transforms import AddOrthonormalNodeIdentifiers


@pytest.mark.parametrize("use_laplacian", [True, False])
def test_add_orthonormal_node_identifiers(use_laplacian):
    n = 4  # num_nodes
    x = torch.rand(n, 10)
    edge_index = torch.tensor([[0, 1, 2, 2], [1, 0, 1, 3]])

    # d_p == num_nodes
    data = Data(x=x, edge_index=edge_index)
    transform = AddOrthonormalNodeIdentifiers(n, use_laplacian)
    data = transform(data)

    assert data.node_ids.shape == (n, n)
    actual = data.node_ids.t() @ data.node_ids
    expected = torch.eye(n, n)
    assert_close(actual, expected)

    # d_p > num_nodes
    data = Data(x=x, edge_index=edge_index)
    transform = AddOrthonormalNodeIdentifiers(n + 1, use_laplacian)
    data = transform(data)

    assert data.node_ids.shape == (n, n + 1)
    actual = data.node_ids.t() @ data.node_ids
    expected = F.pad(torch.eye(n, n), (0, 1, 0, 1), value=0.0)
    assert_close(actual, expected)

    # d_p < num_nodes
    data = Data(x=x, edge_index=edge_index)
    transform = AddOrthonormalNodeIdentifiers(n - 1, use_laplacian)
    data = transform(data)

    assert data.node_ids.shape == (n, n - 1)
    actual = (data.node_ids * data.node_ids).sum(dim=1)
    expected = torch.ones(n)
    assert_close(actual, expected)
