import torch

from torch_geometric.datasets import KarateClub
from torch_geometric.nn.models import BigClam


def test_fit_converges_no_nan():
    graph = KarateClub()
    edge_index = graph[0].edge_index

    model = BigClam(edge_index, num_communities=2)

    model.fit(iterations=5, lr=0.1, verbose=False)
    F = model.forward()
    assert torch.isfinite(F).all()
    assert torch.min(F) >= 0
