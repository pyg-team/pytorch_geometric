import torch
from torch_geometric.nn.conv.mesh_conv import MeshConv


def test_mesh_conv():
    x = torch.randn(1, 5, 25)
    edge_index = torch.tensor([[[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
                               [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]]])

    conv = MeshConv(in_channels=5, out_channels=32, bias=True)
    assert conv.__repr__() == 'MeshConv(in_channels=5, out_channels=32, bias=True, hood=5, aggr=add)'
    out = conv(x, edge_index).squeeze(-1)
    assert out.size() == (1, 32, 25)
