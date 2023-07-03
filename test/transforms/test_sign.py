import torch

from torch_geometric.data import Data
from torch_geometric.testing import withPackage
from torch_geometric.transforms import SIGN


@withPackage('torch>=1.12.0')
def test_sign():
    x = torch.ones(5, 3)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 3, 4],
        [1, 0, 3, 2, 4, 3],
    ])
    data = Data(x=x, edge_index=edge_index)

    transform = SIGN(K=2)
    assert str(transform) == 'SIGN(K=2)'

    expected_x1 = torch.tensor([
        [1, 1, 1],
        [1, 1, 1],
        [0.7071, 0.7071, 0.7071],
        [1.4142, 1.4142, 1.4142],
        [0.7071, 0.7071, 0.7071],
    ])
    expected_x2 = torch.ones(5, 3)

    out = transform(data)
    assert len(out) == 4
    assert torch.equal(out.edge_index, edge_index)
    assert torch.allclose(out.x, x)
    assert torch.allclose(out.x1, expected_x1, atol=1e-4)
    assert torch.allclose(out.x2, expected_x2)
