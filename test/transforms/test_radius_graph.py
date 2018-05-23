import torch
from torch_geometric.transforms import RadiusGraph
from torch_geometric.data import Data


def test_radius_graph():
    assert RadiusGraph(1).__repr__() == 'RadiusGraph(r=1)'

    pos = [[0, 0], [1, 0], [2, 0], [0, 1], [-2, 0], [0, -2]]
    pos = torch.tensor(pos, dtype=torch.float)
    data = Data(pos=pos)

    edge_index = RadiusGraph(1)(data).edge_index
    assert edge_index.tolist() == [[0, 0, 1, 1, 2, 3], [1, 3, 0, 2, 1, 0]]
