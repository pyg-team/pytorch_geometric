import torch

from torch_geometric.data import Data
from torch_geometric.transforms import LocalDegreeProfile


def test_target_indegree():
    assert str(LocalDegreeProfile()) == 'LocalDegreeProfile()'

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    x = torch.Tensor([[1], [1], [1], [1]])  # One isolated node.

    expected = torch.tensor([
        [1, 2, 2, 2, 0],
        [2, 1, 1, 1, 0],
        [1, 2, 2, 2, 0],
        [0, 0, 0, 0, 0],
    ], dtype=torch.float)

    data = Data(edge_index=edge_index, num_nodes=x.size(0))
    data = LocalDegreeProfile()(data)
    assert torch.allclose(data.x, expected, atol=1e-2)

    data = Data(edge_index=edge_index, x=x)
    data = LocalDegreeProfile()(data)
    assert torch.allclose(data.x[:, :1], x)
    assert torch.allclose(data.x[:, 1:], expected, atol=1e-2)
