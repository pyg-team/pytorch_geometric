import pytest
import torch

from torch_geometric.nn import GPSConv, SAGEConv
from torch_geometric.typing import SparseTensor


@pytest.mark.parametrize('norm', [None, 'batch_norm', 'layer_norm'])
def test_gps_conv(norm):
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    row, col = edge_index
    adj1 = SparseTensor(row=col, col=row, sparse_sizes=(4, 4))
    adj2 = adj1.to_torch_sparse_csc_tensor()
    batch = torch.tensor([0, 0, 1, 1])

    conv = GPSConv(16, conv=SAGEConv(16, 16), heads=4, norm=norm)
    conv.reset_parameters()
    assert str(conv) == ('GPSConv(16, conv=SAGEConv(16, 16, aggr=mean), '
                         'heads=4)')

    out = conv(x, edge_index)
    assert out.size() == (4, 16)
    assert torch.allclose(conv(x, adj1.t()), out, atol=1e-6)
    assert torch.allclose(conv(x, adj2.t()), out, atol=1e-6)

    out = conv(x, edge_index, batch)
    assert out.size() == (4, 16)
    assert torch.allclose(conv(x, adj1.t(), batch), out, atol=1e-6)
    assert torch.allclose(conv(x, adj2.t(), batch), out, atol=1e-6)
