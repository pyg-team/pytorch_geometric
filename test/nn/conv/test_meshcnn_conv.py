import pytest
import torch
from torch.nn import Linear, ModuleList, Sequential, Sigmoid

from torch_geometric.nn import MeshCNNConv


@pytest.mark.parametrize('in_channels, out_channels', [
    (1, 1),
    (1, 2),
    (8, 3),
    (8, 3),
    (42, 40),
])
def test_meshcnn_conv(in_channels: int, out_channels: int):
    # m = (V, F), shape [|V| x 3, 3 * |F|]
    # The simplest manifold triangular mesh is a tetrahedron
    E_cardinality = 6  # |E|, the number of edges
    x0 = torch.randn(E_cardinality, in_channels)  # X^(k), the prior layer
    edge_index = torch.tensor([[
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5
    ], [
        1, 2, 3, 4, 2, 0, 4, 5, 5, 3, 0, 1, 2, 5, 4, 0, 0, 3, 5, 1, 1, 4, 3, 2
    ]], dtype=torch.int64)

    # in_channels is the `Dim-Out(k)` in torch.nn.conv.MeshCNNConv
    # out_channels is the `Dim-Out(k+1)` in torch.nn.conv.MeshCNNConv
    conv = MeshCNNConv(in_channels, out_channels)

    # Assert right representation (defined by the class's __repr__ method)
    # WARN: For now we do not account for the 5 default kernels in the
    # representation.
    assert str(conv) == f"MeshCNNConv({in_channels}, {out_channels})"

    x1 = conv(x0, edge_index)
    assert x1.size() == (E_cardinality, out_channels)
    # assert determinism
    assert torch.allclose(conv(x0, edge_index), x1)

    # kernels MUST be a ModuleList of length 5.
    # Where kernels[0] is known as W_0^{(k+1)} in MeshCNNConv etc
    kernels = ModuleList([
        Sequential(Linear(in_channels, out_channels), Sigmoid())
        for _ in range(5)
    ])
    with pytest.warns(UserWarning, match="does not have attribute"):
        conv = MeshCNNConv(in_channels, out_channels, kernels)
    # WARN: For now we do not account for the 5 kernels in the
    # representation
    assert str(conv) == f"MeshCNNConv({in_channels}, {out_channels})"
    x1 = conv(x0, edge_index)
    assert x1.size() == (E_cardinality, out_channels)
