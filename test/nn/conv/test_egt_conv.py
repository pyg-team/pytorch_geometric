import pytest
import torch

from torch_geometric.nn import EGTConv
from torch_geometric.utils import to_dense_adj


@pytest.mark.parametrize('edge_update', [True, False])
def test_egt_conv(edge_update):
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    batch = torch.tensor([0, 0, 1, 1])
    edge_attr = to_dense_adj(edge_index, batch,
                             edge_attr=torch.randn(edge_index.size(1), 8))

    conv = EGTConv(16, edge_dim=8, edge_update=edge_update,
                   num_virtual_nodes=4, heads=4)
    conv.reset_parameters()
    assert str(conv) == 'EGTConv(16, heads=4, num_virtual_nodes=4)'

    if edge_update:
        out_x, out_edge_attr = conv(x, edge_attr, batch)
        assert out_x.size() == (4, 16)
        assert out_edge_attr.size() == edge_attr.size()
    else:
        out_x = conv(x, edge_attr, batch)
        assert out_x.size() == (4, 16)
