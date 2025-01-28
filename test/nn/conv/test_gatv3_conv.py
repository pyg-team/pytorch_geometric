import pytest
import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric.nn import GATv3Conv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import Adj, SparseTensor
from torch_geometric.utils import to_torch_csc_tensor


@pytest.mark.parametrize("share_weights", [False, True])
def test_gatv3_conv(share_weights):
    x1 = torch.randn(4, 8)
    x2 = torch.randn(2, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    # Create a CSC adjacency matrix for tests:
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))

    # Instantiate the layer:
    conv = GATv3Conv(
        in_channels=8,
        out_channels=32,
        heads=2,
        share_weights=share_weights,
    )
    assert str(conv) == "GATv3Conv(8, 32, heads=2)"

    # Standard usage:
    out = conv(x1, edge_index)
    assert out.size() == (4, 64)  # (num_nodes, heads * out_channels)
    # Check consistency on repeated calls:
    conv.eval()
    assert torch.allclose(conv(x1, edge_index), out)

    # Check CSC adjacency:
    # (Tensors must often be transposed if needed, similar to GATv2 tests.)
    assert torch.allclose(conv(x1, adj1.to_sparse_coo()), out, atol=1e-6)

    # If torch_sparse is available, also check with a SparseTensor:
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        s_adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        assert torch.allclose(conv(x1, s_adj.t()), out, atol=1e-6)

    # Test scripting/JIT if in "full test" mode:
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
            assert torch.allclose(jit(x1, s_adj.t()), out, atol=1e-6)

    # Test return_attention_weights:
    result = conv(x1, edge_index, return_attention_weights=True)
    out_att, (ei_att, alpha_att) = result
    assert torch.allclose(out_att, out)
    assert ei_att.size(0) == 2 and ei_att.size(1) == edge_index.size(1) + 4
    # Typically, GATv3 might add self-loops automatically, so the edge count
    # may differ. Adjust the check if you remove that or keep it.

    assert alpha_att.size() == (ei_att.size(1), 2)  # (num_edges, heads)
    assert alpha_att.min() >= 0 and alpha_att.max() <= 1

    # Bipartite message passing:
    # Use an adjacency shaped for (src_nodes=4, dst_nodes=2).
    adj_bip = to_torch_csc_tensor(edge_index, size=(4, 2))

    out_bip = conv((x1, x2), edge_index)
    assert out_bip.size() == (2, 64)
    assert torch.allclose(conv((x1, x2), edge_index), out_bip)
    assert torch.allclose(conv((x1, x2), adj_bip.t()), out_bip, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        s_adj_bip = SparseTensor.from_edge_index(edge_index,
                                                 sparse_sizes=(4, 2))
        assert torch.allclose(conv((x1, x2), s_adj_bip.t()), out_bip,
                              atol=1e-6)

    if is_full_test():

        class MyBipModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = conv

            def forward(self, x_tuple, edge_index: Adj) -> Tensor:
                return self.conv(x_tuple, edge_index)

        jit = torch.jit.script(MyBipModule())
        assert torch.allclose(jit((x1, x2), edge_index), out_bip)


def test_gatv3_conv_with_edge_attr():
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 1, 1]])
    edge_weight = torch.randn(edge_index.size(1))  # 1D features
    edge_attr = torch.randn(edge_index.size(1), 4)  # 4D features

    # Case 1: 1D edge features with fill_value=0.5
    conv = GATv3Conv(
        in_channels=8,
        out_channels=32,
        heads=2,
        edge_dim=1,
        fill_value=0.5,
    )
    out = conv(x, edge_index, edge_weight)
    assert out.size() == (4, 64)

    # Case 2: 1D edge features with fill_value='mean'
    conv = GATv3Conv(
        in_channels=8,
        out_channels=32,
        heads=2,
        edge_dim=1,
        fill_value="mean",
    )
    out = conv(x, edge_index, edge_weight)
    assert out.size() == (4, 64)

    # Case 3: 4D edge features with fill_value=0.5
    conv = GATv3Conv(
        in_channels=8,
        out_channels=32,
        heads=2,
        edge_dim=4,
        fill_value=0.5,
    )
    out = conv(x, edge_index, edge_attr)
    assert out.size() == (4, 64)

    # Case 4: 4D edge features with fill_value='mean'
    conv = GATv3Conv(
        in_channels=8,
        out_channels=32,
        heads=2,
        edge_dim=4,
        fill_value="mean",
    )
    out = conv(x, edge_index, edge_attr)
    assert out.size() == (4, 64)
