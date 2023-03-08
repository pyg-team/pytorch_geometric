import torch

from torch_geometric.data import Data
from torch_geometric.testing import withPackage
from torch_geometric.transforms import RadiusGraph
from torch_geometric.utils import coalesce


@withPackage('torch_cluster')
def test_radius_graph():
    assert str(RadiusGraph(r=1)) == 'RadiusGraph(r=1)'

    pos = torch.Tensor([[0, 0], [1, 0], [2, 0], [0, 1], [-2, 0], [0, -2]])

    data = Data(pos=pos)
    data = RadiusGraph(r=1.5)(data)
    assert len(data) == 2
    assert data.pos.tolist() == pos.tolist()
    assert coalesce(data.edge_index).tolist() == [[0, 0, 1, 1, 1, 2, 3, 3],
                                                  [1, 3, 0, 2, 3, 1, 0, 1]]
