import torch
from torch_geometric.transforms import RadiusGraph
from torch_geometric.data import Data


def test_radius_graph():
    assert RadiusGraph(r=1).__repr__() == 'RadiusGraph(r=1)'

    pos = torch.Tensor([[0, 0], [1, 0], [2, 0], [0, 1], [-2, 0], [0, -2]])

    data = Data(pos=pos)
    data = RadiusGraph(r=1)(data)
    assert len(data) == 2
    assert data.pos.tolist() == pos.tolist()
    assert data.edge_index.tolist() == [[0, 0, 1, 1, 2, 3], [1, 3, 0, 2, 1, 0]]
