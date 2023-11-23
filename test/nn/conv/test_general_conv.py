import pytest
import torch

import torch_geometric.typing
from torch_geometric.nn import GeneralConv
from torch_geometric.typing import SparseTensor


@pytest.mark.parametrize('kwargs', [
    dict(),
    dict(skip_linear=True),
    dict(directed_msg=False),
    dict(heads=3),
    dict(attention=True),
    dict(heads=3, attention=True),
    dict(heads=3, attention=True, attention_type='dot_product'),
    dict(l2_normalize=True),
])
def test_general_conv(kwargs):
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_attr = torch.randn(edge_index.size(1), 16)

    conv = GeneralConv(8, 32, in_edge_channels=16, **kwargs)
    assert str(conv) == 'GeneralConv(8, 32)'
    out = conv(x, edge_index, edge_attr)
    assert out.size() == (4, 32)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, edge_attr, (4, 4))
        assert torch.allclose(conv(x, adj.t()), out, atol=1e-6)
