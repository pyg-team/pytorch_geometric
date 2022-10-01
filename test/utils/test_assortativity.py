import pytest
import torch

from torch_geometric.utils import assortativity, coalesce


def test_assortativity():
    torch.manual_seed(12345)

    num_nodes = 100
    num_edges = 1000
    edge_index = torch.randint(0, num_nodes, size=(2, num_edges))
    edge_index = coalesce(edge_index)
    out = assortativity(edge_index)
    assert pytest.approx(out, abs=1e-5) == -0.05669079348444939
