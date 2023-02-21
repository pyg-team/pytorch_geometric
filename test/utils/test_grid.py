import torch

from torch_geometric.testing import is_full_test
from torch_geometric.utils import grid


def test_grid():
    (row, col), pos = grid(height=3, width=2)

    expected_row = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    expected_col = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5]
    expected_row += [3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
    expected_col += [0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5]

    expected_pos = [[0, 2], [1, 2], [0, 1], [1, 1], [0, 0], [1, 0]]

    assert row.tolist() == expected_row
    assert col.tolist() == expected_col
    assert pos.tolist() == expected_pos

    if is_full_test():
        jit = torch.jit.script(grid)
        (row, col), pos = jit(height=3, width=2)
        assert row.tolist() == expected_row
        assert col.tolist() == expected_col
        assert pos.tolist() == expected_pos
