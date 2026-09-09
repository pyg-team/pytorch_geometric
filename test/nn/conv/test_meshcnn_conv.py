import pytest
import torch
from torch.nn import Linear, ModuleList, Sequential, Sigmoid

from torch_geometric.nn import MeshCNNConv
from torch_geometric.testing import withDevice

TETRAHEDRON_EDGE_INDEX = torch.tensor(
    [[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5],
     [1, 2, 3, 4, 2, 0, 4, 5, 5, 3, 0, 1, 2, 5, 4, 0, 0, 3, 5, 1, 1, 4, 3, 2]],
    dtype=torch.int64)


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


def test_meshcnn_conv_message_preserves_dtype():
    # `message()` used to allocate its output buffer via a bare
    # `torch.empty(E4, out_channels)`, which always defaults to float32.
    # Assigning float64 kernel outputs into that buffer silently downcasts
    # them, losing precision. This is invisible from `forward()`'s final
    # output because `update()`'s `kernel(x) + inputs` addition promotes
    # float32 + float64 back to float64, masking the loss -- so this test
    # calls `message()` directly, where the bug actually lives.
    conv = MeshCNNConv(in_channels=8, out_channels=3).double()
    x_j = torch.randn(TETRAHEDRON_EDGE_INDEX.size(1), 8, dtype=torch.float64)

    out = conv.message(x_j)
    assert out.dtype == torch.float64


@withDevice
def test_meshcnn_conv_device_propagation(device):
    # `message()`'s bare `torch.empty(...)` also always allocates on CPU,
    # ignoring `x_j.device`. On any non-CPU device this crashes `forward()`
    # with a device-mismatch error.
    x = torch.randn(6, 8, device=device)
    edge_index = TETRAHEDRON_EDGE_INDEX.to(device)
    conv = MeshCNNConv(in_channels=8, out_channels=3).to(device)

    out = conv(x, edge_index)
    assert out.device == x.device
    assert out.size() == (6, 3)


def test_meshcnn_conv_reset_parameters():
    torch.manual_seed(0)
    conv = MeshCNNConv(in_channels=4, out_channels=4)
    before = [kernel.weight.clone() for kernel in conv.kernels]

    conv.reset_parameters()

    for kernel, weight_before in zip(conv.kernels, before):
        assert not torch.allclose(kernel.weight, weight_before)
