import torch

from torch_geometric.data import Data
from torch_geometric.transforms import Cartesian


def test_cartesian():
    assert str(Cartesian()) == 'Cartesian(norm=True, max_value=None)'

    pos = torch.tensor([[-1.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_attr = torch.tensor([1.0, 2.0, 3.0, 4.0])

    data = Data(edge_index=edge_index, pos=pos)
    data = Cartesian(norm=False)(data)
    assert len(data) == 3
    assert torch.equal(data.pos, pos)
    assert torch.equal(data.edge_index, edge_index)
    assert torch.allclose(
        data.edge_attr,
        torch.tensor([[-1.0, 0.0], [1.0, 0.0], [-2.0, 0.0], [2.0, 0.0]]),
    )

    data = Data(edge_index=edge_index, pos=pos, edge_attr=edge_attr)
    data = Cartesian(norm=True)(data)
    assert len(data) == 3
    assert torch.equal(data.pos, pos)
    assert torch.equal(data.edge_index, edge_index)
    assert torch.allclose(
        data.edge_attr,
        torch.tensor([
            [1, 0.25, 0.5],
            [2, 0.75, 0.5],
            [3, 0, 0.5],
            [4, 1, 0.5],
        ]),
    )
