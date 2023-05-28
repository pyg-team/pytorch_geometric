import torch

from torch_geometric.testing import is_full_test
from torch_geometric.utils import get_laplacian


def test_get_laplacian():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_weight = torch.tensor([1, 2, 2, 4], dtype=torch.float)

    lap = get_laplacian(edge_index, edge_weight)
    assert lap[0].tolist() == [[0, 1, 1, 2, 0, 1, 2], [1, 0, 2, 1, 0, 1, 2]]
    assert lap[1].tolist() == [-1, -2, -2, -4, 1, 4, 4]

    if is_full_test():
        jit = torch.jit.script(get_laplacian)
        lap = jit(edge_index, edge_weight)
        assert lap[0].tolist() == [[0, 1, 1, 2, 0, 1, 2],
                                   [1, 0, 2, 1, 0, 1, 2]]
        assert lap[1].tolist() == [-1, -2, -2, -4, 1, 4, 4]

    lap_sym = get_laplacian(edge_index, edge_weight, normalization='sym')
    assert lap_sym[0].tolist() == lap[0].tolist()
    assert lap_sym[1].tolist() == [-0.5, -1, -0.5, -1, 1, 1, 1]

    lap_rw = get_laplacian(edge_index, edge_weight, normalization='rw')
    assert lap_rw[0].tolist() == lap[0].tolist()
    assert lap_rw[1].tolist() == [-1, -0.5, -0.5, -1, 1, 1, 1]
