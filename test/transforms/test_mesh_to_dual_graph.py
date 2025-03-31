import torch

from torch_geometric.data import Data
from torch_geometric.transforms import MeshToDualGraph


def test_square_to_dual() -> None:
    transform = MeshToDualGraph()
    assert str(transform) == 'MeshToDualGraph()'

    square_pos = torch.tensor([
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
    ])

    square_face = torch.tensor([
        [0, 2, 6, 4, 2, 7],
        [1, 3, 7, 5, 6, 3],
        [3, 7, 5, 1, 4, 1],
        [2, 6, 4, 0, 0, 5],
    ])

    square = Data(pos=square_pos, face=square_face)

    dual_actual_pos = torch.tensor([
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    dual_actual_edge_index = torch.tensor([
        [
            1,
            0,
            2,
            1,
            3,
            2,
            3,
            0,
            4,
            1,
            4,
            2,
            4,
            3,
            4,
            0,
            5,
            1,
            5,
            0,
            5,
            3,
            5,
            2,
        ],
        [
            0,
            1,
            1,
            2,
            2,
            3,
            0,
            3,
            1,
            4,
            2,
            4,
            3,
            4,
            0,
            4,
            1,
            5,
            0,
            5,
            3,
            5,
            2,
            5,
        ],
    ])

    dual_actual = Data(pos=dual_actual_pos, edge_index=dual_actual_edge_index)

    dual = transform.forward(square)

    assert dual.pos.tolist() == dual_actual.pos.tolist()
    assert dual.edge_index.tolist() == dual_actual.edge_index.tolist()
