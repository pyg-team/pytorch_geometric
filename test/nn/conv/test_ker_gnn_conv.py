import pytest
import torch

from torch_geometric.nn import KerGNNConv
from torch_geometric.utils import to_torch_csc_tensor


@pytest.mark.parametrize('kernel', ['rw', 'drw'])
def test_ker_gnn_conv(kernel):
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))

    conv = KerGNNConv(16, 32, kernel=kernel, power=2)
    assert str(conv) == f'KerGNNConv(16, 32, kernel={kernel}, power=2)'

    out1 = conv(x, edge_index)
    assert out1.size() == (4, 32)
    assert torch.allclose(conv(x, adj1.t()), out1, atol=1e-6)
