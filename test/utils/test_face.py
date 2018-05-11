import torch
from torch_geometric.utils import face_to_edge_index


def test_face_to_edge_index():
    face = torch.tensor([[0, 0], [1, 1], [2, 3]])
    row, col = face_to_edge_index(face)

    expected_row = [0, 0, 0, 1, 1, 1, 2, 2, 3, 3]
    expected_col = [1, 2, 3, 0, 2, 3, 0, 1, 0, 1]

    assert row.tolist() == expected_row
    assert col.tolist() == expected_col
