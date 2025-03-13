import torch

from torch_geometric.nn.models import SGFormer


def test_sgformer():
    x = torch.randn(10, 16)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    ])
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    model = SGFormer(
        in_channels=16,
        hidden_channels=128,
        out_channels=40,
    )
    out = model(x, edge_index, batch)
    assert out.size() == (10, 40)
