import torch
from torch_geometric.transforms import Spherical
from torch_geometric.data import Data


def test_spherical():
    assert Spherical().__repr__() == 'Spherical(norm=True, max_value=None)'

    pos = torch.tensor([[0, 0, 0], [0, 1, 1]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    data = Data(edge_index=edge_index, pos=pos)
    print(data)

    # out = Spherical()(data).edge_attr.view(-1).tolist()
    # assert approx(out) == [1, 0.25, 0, 1, 0.75, 1]

    # data.edge_attr = torch.tensor([1, 1], dtype=torch.float)
    # out = Spherical()(data).edge_attr.view(-1).tolist()
    # assert approx(out) == [1, 1, 0.25, 0, 1, 1, 0.75, 1]
