import pytest
import torch

from torch_geometric.nn.models import EGT


@pytest.mark.parametrize('edge_update', [True, False])
def test_egt(edge_update):
    x = torch.randn(10, 16)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    ])
    edge_attr = torch.randn(edge_index.size(1), 16)

    model = EGT(
        node_channels=16,
        edge_channels=16,
        out_channels=40,
        edge_update=edge_update,
        num_layers=3,
        num_heads=4,
        dropout=0.3,
    )

    out = model(x, edge_index, edge_attr)
    assert out.size() == (10, 40)
