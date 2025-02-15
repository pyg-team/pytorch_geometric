import pytest
import torch

import torch_geometric.typing
from torch_geometric.nn import GMMConv
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_torch_coo_tensor


@pytest.mark.parametrize('separate_gaussians', [True, False])
def test_gmm_conv(separate_gaussians):
    x1 = torch.randn(4, 8)
    x2 = torch.randn(2, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    value = torch.rand(edge_index.size(1), 3)
    adj1 = to_torch_coo_tensor(edge_index, value, size=(4, 4))

    conv = GMMConv(8, 32, dim=3, kernel_size=25,
                   separate_gaussians=separate_gaussians)
    assert str(conv) == 'GMMConv(8, 32, dim=3)'
    out = conv(x1, edge_index, value)
    assert out.size() == (4, 32)
    assert torch.allclose(conv(x1, edge_index, value, size=(4, 4)), out)
    # t() expects a tensor with <= 2 sparse and 0 dense dimensions
    assert torch.allclose(conv(x1, adj1.transpose(0, 1).coalesce()), out)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, value, (4, 4))
        assert torch.allclose(conv(x1, adj2.t()), out)

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit(x1, edge_index, value), out)
        assert torch.allclose(jit(x1, edge_index, value, size=(4, 4)), out)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit(x1, adj2.t()), out)

    # Test bipartite message passing:
    adj1 = to_torch_coo_tensor(edge_index, value, size=(4, 2))

    conv = GMMConv((8, 16), 32, dim=3, kernel_size=5,
                   separate_gaussians=separate_gaussians)
    assert str(conv) == 'GMMConv((8, 16), 32, dim=3)'

    out1 = conv((x1, x2), edge_index, value)
    assert out1.size() == (2, 32)
    assert torch.allclose(conv((x1, x2), edge_index, value, (4, 2)), out1)
    assert torch.allclose(conv((x1, x2),
                               adj1.transpose(0, 1).coalesce()), out1)

    out2 = conv((x1, None), edge_index, value, (4, 2))
    assert out2.size() == (2, 32)
    assert torch.allclose(conv((x1, None),
                               adj1.transpose(0, 1).coalesce()), out2)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, value, (4, 2))
        assert torch.allclose(conv((x1, x2), adj2.t()), out1)
        assert torch.allclose(conv((x1, None), adj2.t()), out2)

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit((x1, x2), edge_index, value), out1)
        assert torch.allclose(jit((x1, x2), edge_index, value, size=(4, 2)),
                              out1)
        assert torch.allclose(jit((x1, None), edge_index, value, size=(4, 2)),
                              out2)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit((x1, x2), adj2.t()), out1)
            assert torch.allclose(jit((x1, None), adj2.t()), out2)


@pytest.mark.parametrize('separate_gaussians', [True, False])
def test_lazy_gmm_conv(separate_gaussians):
    x1 = torch.randn(4, 8)
    x2 = torch.randn(2, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    value = torch.rand(edge_index.size(1), 3)

    conv = GMMConv(-1, 32, dim=3, kernel_size=25,
                   separate_gaussians=separate_gaussians)
    assert str(conv) == 'GMMConv(-1, 32, dim=3)'
    out = conv(x1, edge_index, value)
    assert out.size() == (4, 32)

    conv = GMMConv((-1, -1), 32, dim=3, kernel_size=25,
                   separate_gaussians=separate_gaussians)
    assert str(conv) == 'GMMConv((-1, -1), 32, dim=3)'
    out = conv((x1, x2), edge_index, value)
    assert out.size() == (2, 32)
