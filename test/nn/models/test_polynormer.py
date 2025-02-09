import torch

from torch_geometric.nn.models import Polynormer


def test_polynormer():
    x = torch.ones(10, 16, dtype=torch.float32)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    ])

    model = Polynormer(
        in_channels=16,
        hidden_channels=128,
        out_channels=40,
    )
    out = model(x, edge_index)
    assert out.size() == (10, 40)
