import copy

import torch

from torch_geometric.data import Data
from torch_geometric.transforms import SIGN


def test_sign_transform():
    assert SIGN(K=1).__repr__() == 'SIGN(K=1)'
    edge_index = torch.tensor([
        [0, 1, 2, 3, 3, 4],
        [1, 0, 3, 2, 4, 3],
    ])
    x = torch.ones(5, 3)
    expected_x1 = torch.tensor([[1, 1, 1], [1, 1, 1], [0.7071, 0.7071, 0.7071],
                                [1.4142, 1.4142, 1.4142],
                                [0.7071, 0.7071, 0.7071]])
    expected_x2 = torch.ones(5, 3)
    data = Data(x=x, edge_index=edge_index, num_nodes=5)
    transform = SIGN(K=2)
    transformed_data = transform(copy.copy(data))
    assert len(transformed_data) == 5
    assert transformed_data.edge_index.tolist() == edge_index.tolist()
    assert transformed_data.x.tolist() == x.tolist()
    assert torch.allclose(transformed_data.x1, expected_x1, atol=1e-4)
    assert torch.allclose(transformed_data.x2, expected_x2, atol=1e-4)
