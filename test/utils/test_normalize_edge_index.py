import pytest
import torch

from torch_geometric.utils import normalize_edge_index


@pytest.mark.parametrize('add_self_loops', [False, True])
@pytest.mark.parametrize('symmetric', [False, True])
def test_normalize_edge_index(add_self_loops: bool, symmetric: bool):
    edge_index = torch.tensor([[0, 2, 2, 3], [2, 0, 3, 0]])

    out = normalize_edge_index(
        edge_index,
        add_self_loops=add_self_loops,
        symmetric=symmetric,
    )
    assert isinstance(out, tuple) and len(out) == 2
    if not add_self_loops:
        assert out[0].equal(edge_index)
    else:
        assert out[0].tolist() == [
            [0, 2, 2, 3, 0, 1, 2, 3],
            [2, 0, 3, 0, 0, 1, 2, 3],
        ]

    assert out[1].min() >= 0.0
    assert out[1].min() <= 1.0
