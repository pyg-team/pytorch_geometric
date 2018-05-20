import torch
from torch_geometric.transform import RadiusGraph
from torch_geometric.datasets.dataset import Data


def test_radius_graph():
    assert RadiusGraph(1).__repr__() == 'RadiusGraph(r=1)'

    pos = [[0, 0], [1, 0], [2, 0], [0, 1], [-2, 0], [0, -2]]
    pos = torch.tensor(pos, dtype=torch.float)
    data = Data(None, pos, None, None, None)

    edge_index = RadiusGraph(1)(data).index
    assert edge_index.tolist() == [[0, 0, 1, 1, 2, 3], [1, 3, 0, 2, 1, 0]]
