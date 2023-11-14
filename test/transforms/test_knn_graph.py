import torch

from torch_geometric.data import Data
from torch_geometric.testing import withPackage
from torch_geometric.transforms import KNNGraph


@withPackage('torch_cluster')
def test_knn_graph():
    assert str(KNNGraph()) == 'KNNGraph(k=6)'

    pos = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [-2.0, 0.0],
        [0.0, -2.0],
    ])

    expected_row = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5]
    expected_col = [1, 2, 3, 4, 5, 0, 2, 3, 5, 0, 1, 0, 1, 4, 0, 3, 0, 1]

    data = Data(pos=pos)
    data = KNNGraph(k=2, force_undirected=True)(data)
    assert len(data) == 2
    assert data.pos.tolist() == pos.tolist()
    assert data.edge_index[0].tolist() == expected_row
    assert data.edge_index[1].tolist() == expected_col
