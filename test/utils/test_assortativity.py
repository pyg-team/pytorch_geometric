import pytest
import torch

from torch_geometric.utils import assortativity


def test_assortativity():
    # completely assortative graph
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5],
                               [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2, 5, 4]])
    out = assortativity(edge_index)
    assert pytest.approx(out, abs=1e-5) == 1.0

    # completely disassortative graph
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 5, 5, 5, 5],
                               [5, 5, 5, 5, 5, 0, 1, 2, 3, 4]])
    out = assortativity(edge_index)
    assert pytest.approx(out, abs=1e-5) == -1.0
