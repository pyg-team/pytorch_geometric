import torch

from torch_geometric.nn import FusedGATConv
from torch_geometric.testing import withCUDA, withPackage


@withCUDA
@withPackage('dgNN')
def test_fused_gat_conv():
    device = torch.device('cuda')

    x = torch.randn(4, 8, device=device)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], device=device)

    csr, csc, perm = FusedGATConv.to_graph_format(edge_index, size=(4, 4))

    conv = FusedGATConv(8, 32, heads=2, add_self_loops=False).to(device)
    assert str(conv) == 'FusedGATConv(8, 32, heads=2)'

    out = conv(x, csr, csc, perm)
    assert out.size() == (4, 64)
