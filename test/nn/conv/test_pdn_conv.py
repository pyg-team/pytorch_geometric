import torch

import torch_geometric.typing
from torch_geometric.nn import PDNConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor


def test_pdn_conv():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_attr = torch.randn(edge_index.size(1), 8)

    conv = PDNConv(16, 32, edge_dim=8, hidden_channels=128)
    assert str(conv) == "PDNConv(16, 32)"

    out = conv(x, edge_index, edge_attr)
    assert out.size() == (4, 32)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, edge_attr, (4, 4))
        assert torch.allclose(conv(x, adj.t()), out, atol=1e-6)

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit(x, edge_index, edge_attr), out)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit(x, adj.t()), out, atol=1e-6)


def test_pdn_conv_with_sparse_node_input_feature():
    x = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 0], [0, 1]]),
        values=torch.tensor([1.0, 1.0]),
        size=torch.Size([4, 16]),
    )
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_attr = torch.randn(edge_index.size(1), 8)

    conv = PDNConv(16, 32, edge_dim=8, hidden_channels=128)

    out = conv(x, edge_index, edge_attr)
    assert out.size() == (4, 32)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, edge_attr, (4, 4))
        assert torch.allclose(conv(x, adj.t(), edge_attr), out, atol=1e-6)

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit(x, edge_index, edge_attr), out)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit(x, adj.t(), edge_attr), out, atol=1e-6)
