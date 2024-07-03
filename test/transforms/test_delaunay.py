import torch

from torch_geometric.data import Data
from torch_geometric.testing import withPackage
from torch_geometric.transforms import Delaunay


@withPackage('scipy')
def test_delaunay():
    assert str(Delaunay()) == 'Delaunay()'

    pos = torch.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]], dtype=torch.float)
    data = Data(pos=pos)
    data = Delaunay()(data)
    assert len(data) == 2
    assert data.face.tolist() == [[3, 1], [1, 3], [0, 2]]

    pos = torch.tensor([[-1, -1], [-1, 1], [1, 1]], dtype=torch.float)
    data = Data(pos=pos)
    data = Delaunay()(data)
    assert len(data) == 2
    assert data.face.tolist() == [[0], [1], [2]]

    pos = torch.tensor([[-1, -1], [1, 1]], dtype=torch.float)
    data = Data(pos=pos)
    data = Delaunay()(data)
    assert len(data) == 2
    assert data.edge_index.tolist() == [[0, 1], [1, 0]]

    pos = torch.tensor([[-1, -1]], dtype=torch.float)
    data = Data(pos=pos)
    data = Delaunay()(data)
    assert len(data) == 2
    assert data.edge_index.tolist() == [[], []]
