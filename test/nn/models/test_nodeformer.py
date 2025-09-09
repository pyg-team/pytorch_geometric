import torch

from torch_geometric.nn.models import NodeFormer


def test_nodeformer():
    x = torch.randn(10, 16)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    ])

    model = NodeFormer(
        in_channels=16,
        hidden_channels=128,
        out_channels=40,
        num_layers=3,
    )
    out, link_loss = model(x, edge_index)
    assert out.size() == (10, 40)
    assert len(link_loss) == 3
