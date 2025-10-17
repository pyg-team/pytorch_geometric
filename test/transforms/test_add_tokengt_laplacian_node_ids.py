import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from torch_geometric.data import Data
from torch_geometric.transforms import AddTokenGTLaplacianNodeIds


def test_add_token_gt_laplacian_node_ids():
    n = 4  # num_nodes
    x = torch.rand(n, 10)
    edge_index = torch.tensor([[0, 1, 2, 2], [1, 0, 1, 3]])

    # d_p == num_nodes
    data = Data(x=x, edge_index=edge_index)
    transform = AddTokenGTLaplacianNodeIds(n)
    data = transform(data)

    assert data.node_ids.shape == (n, n)
    actual = data.node_ids.t() @ data.node_ids
    expected = torch.eye(n, n)
    assert_close(actual, expected)

    # d_p > num_nodes
    data = Data(x=x, edge_index=edge_index)
    transform = AddTokenGTLaplacianNodeIds(n + 1)
    data = transform(data)

    assert data.node_ids.shape == (n, n + 1)
    actual = data.node_ids.t() @ data.node_ids
    expected = F.pad(torch.eye(n, n), (0, 1, 0, 1), value=0.0)
    assert_close(actual, expected)

    # d_p < num_nodes
    data = Data(x=x, edge_index=edge_index)
    transform = AddTokenGTLaplacianNodeIds(n - 1)
    data = transform(data)

    assert data.node_ids.shape == (n, n - 1)
    actual = data.node_ids.t() @ data.node_ids
    expected = torch.eye(n - 1, n - 1)
    assert_close(actual, expected)
