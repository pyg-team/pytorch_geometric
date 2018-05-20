import torch
from torch_geometric.transform import NNGraph
from torch_geometric.datasets.dataset import Data


def test_nn_graph():
    assert NNGraph().__repr__() == 'NNGraph(k=6)'

    pos = [[0, 0], [1, 0], [2, 0], [0, 1], [-2, 0], [0, -2]]
    pos = torch.tensor(pos, dtype=torch.float)
    data = Data(None, pos, None, None, None)

    row, col = NNGraph(2)(data).index
    expected_row = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5]
    expected_col = [1, 2, 3, 4, 5, 0, 2, 3, 5, 0, 1, 0, 1, 4, 0, 3, 0, 1]

    assert row.tolist() == expected_row
    assert col.tolist() == expected_col
