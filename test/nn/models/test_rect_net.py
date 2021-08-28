import torch
from torch_geometric.nn.models import RECT_L


def test_rect_net():
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_attr = torch.tensor([0.3395, 0.0025, 0.0014, 0.0033])
    out_channels = 8
    model = RECT_L(in_feats=8, n_hidden=16, dropout=0.0)
    assert str(model) == 'RECT_L(8, 8)'
    assert model(x, edge_index, edge_attr).size() == (3, out_channels)
