import pytest
import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric.nn import GATv3Conv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import Adj, SparseTensor
from torch_geometric.utils import to_torch_csc_tensor


@pytest.mark.parametrize('share_weights', [False, True])
@pytest.mark.parametrize('residual', [False, True])
def test_gatv3_conv(share_weights, residual):
    x1 = torch.randn(4, 8)
    x2 = torch.randn(2, 8)
    # The (0,0) edge is a self-loop that will be removed and re-added.
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [0, 0, 1, 1, 0]])
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))

    conv = GATv3Conv(8, 32, heads=2, share_weights=share_weights,
                     residual=residual)
    assert str(conv) == 'GATv3Conv(8, 32, heads=2)'
    out = conv(x1, edge_index)
    assert out.size() == (4, 64)
    assert torch.allclose(conv(x1, edge_index), out)
    assert torch.allclose(conv(x1, adj1.t()), out, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert torch.allclose(conv(x1, adj2.t()), out, atol=1e-6)

    if is_full_test():

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = conv

            def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
                return self.conv(x, edge_index)

        jit = torch.jit.script(MyModule())
        assert torch.allclose(jit(x1, edge_index), out)
        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit(x1, adj2.t()), out, atol=1e-6)

    # Test `return_attention_weights`.
    result = conv(x1, edge_index, return_attention_weights=True)
    out_att, (ei_att, alpha_att) = result
    assert torch.allclose(out_att, out)
    # Corrected Assertion:
    # + 4 new self-loops = 7 total edges.
    assert ei_att.size() == (2, 7)
    assert alpha_att.size() == (7, 2)
    assert alpha_att.min() >= 0 and alpha_att.max() <= 1

    # Bipartite message passing.
    # Self-loops are not added in the bipartite case.
    bipartite_edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    conv_bipartite = GATv3Conv((8, 8), 32, heads=2,
                               share_weights=share_weights, residual=residual,
                               add_self_loops=False)
    out_bip = conv_bipartite((x1, x2), bipartite_edge_index)
    assert out_bip.size() == (2, 64)
    assert torch.allclose(conv_bipartite((x1, x2), bipartite_edge_index),
                          out_bip)


def test_gatv3_conv_with_edge_attr():
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 1, 1]])
    edge_attr = torch.randn(edge_index.size(1), 4)

    # Test with 4D edge features.
    conv = GATv3Conv(8, 32, heads=2, edge_dim=4, fill_value=0.5)
    out = conv(x, edge_index, edge_attr)
    assert out.size() == (4, 64)

    # Test with 1D edge features, which should be reshaped.
    conv = GATv3Conv(8, 32, heads=2, edge_dim=1, fill_value='mean')
    out = conv(x, edge_index, edge_attr[:, :1])
    assert out.size() == (4, 64)
