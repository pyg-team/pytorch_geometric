import pytest
import torch

import torch_geometric.typing
from torch_geometric.contrib.nn.conv.res_graph_conv import (
    ResGConv,
    res_gconv_norm,
)
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    to_edge_index,
    to_torch_csr_tensor,
)


def test_res_gconv_norm():
    # we first make a minimal example for all cases of the
    # sparse tensors

    # this is for the general case if both
    # isinstance(edge_index, SparseTensor) and
    # is_torch_sparse_tensor(edge_index)
    # are false
    edge_index = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]])
    edge_weight = torch.tensor([1.0, 1.0, 1.0, 1.0])
    assert not is_torch_sparse_tensor(edge_index)
    expected_edge_index = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]])
    expected_edge_weight = 0.5 * torch.tensor([1.0, 1.0, 1.0, 1.0])
    edge_index_out, edge_weight_out = res_gconv_norm(edge_index, edge_weight,
                                                     add_self_loops=False)
    assert torch.equal(edge_index_out, expected_edge_index)
    assert torch.allclose(expected_edge_weight, edge_weight_out)

    # this is for the case where
    # edge_index passed to res_gconv_norm
    # isinstance(edge_index, SparseTensor)
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(2, 2))
        expected_adj = 0.5 * torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        adj_out = res_gconv_norm(adj, add_self_loops=False)
        assert torch.equal(adj_out.t().to_dense(), expected_adj)

    # this is for the case where
    # edge_index pass to res_gconv_norm
    # is_torch_sparse_tensor(edge_index)
    adj = to_torch_csr_tensor(edge_index)
    expected_edge_index = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]])
    expected_edge_weight = 0.5 * torch.tensor([1.0, 1.0, 1.0, 1.0])
    adj_out, none = res_gconv_norm(adj, add_self_loops=False)
    edge_index_out, edge_weight_out, to_edge_index(adj_out)
    assert torch.equal(edge_index_out, expected_edge_index)
    assert torch.equal(edge_weight_out, expected_edge_weight)


@pytest.fixture
def res_gconv_factory():
    def _create_layer(channels: int = 1, shared_weights: bool = False,
                      add_self_loops: bool = False, cached: bool = False,
                      normalize: bool = True, **kwargs):
        """Fixture providing a res_gconv layer with fixed
        weights for testing.
        """
        layer = ResGConv(channels=channels, shared_weights=shared_weights,
                         add_self_loops=add_self_loops, cached=cached,
                         normalize=normalize, **kwargs)
        with torch.no_grad():
            layer.U.weight.fill_(1.0)
            if layer.U.bias is not None:
                layer.U.bias.zero_()
            layer.W.weight.fill_(1.0)
            if layer.W.bias is not None:
                layer.W.bias.zero_()
        return layer

    return _create_layer


def test_res_gconv(res_gconv_factory):
    # test the branch where
    # isinstance(edge_index, Tensor)
    #
    # we setup a dummy input where the first index
    # denotes the number of nodes in the graph (2)
    # and the second one denotes the hidden dimension
    # the res_gconv layer expects (1) from the fixture
    # above. First let's test it for the unnormalized
    # case, where the output should simply be
    # :math:`\mathbf{XU} + \mathbf{AXW}`
    layer = res_gconv_factory(normalize=False)
    x = torch.ones(2, 1)
    edge_index = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]])
    edge_weight = torch.tensor([1.0, 1.0, 1.0, 1.0])
    out1 = layer(x, edge_index, edge_weight)
    out2 = layer(x, edge_index)
    # test against the output calculated by hand
    assert torch.allclose(out1, 3.0 * x)
    assert torch.allclose(out2, 3.0 * x)
    # let's do the same using the normalized option
    # such that the output should now be
    # :math:`\mathbf{XU} + \mathbf{\hat{A}XW}`
    layer = res_gconv_factory()
    x = torch.ones(2, 1)
    edge_index = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]])
    edge_weight = torch.tensor([1.0, 1.0, 1.0, 1.0])
    out1 = layer(x, edge_index, edge_weight)
    out2 = layer(x, edge_index, edge_weight)
    assert torch.allclose(out1, 2.0 * x)
    assert torch.allclose(out2, 2.0 * x)

    # test the branch where
    # isinstance(edge_index, SparseTensor)
    #
    # we setup a dummy input where the first index
    # denotes the number of nodes in the graph (2)
    # and the second one denotes the hidden dimension
    # the res_gconv layer expects (1) from the fixture
    # above. First let's test it for the unnormalized
    # case, where the output should simply be
    # :math:`\mathbf{XU} + \mathbf{AXW}`
    layer = res_gconv_factory(normalize=False)
    x = torch.ones(2, 1)
    edge_index = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]])
    adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(2, 2))
    edge_weight = torch.tensor([1.0, 1.0, 1.0, 1.0])
    out1 = layer(x, adj, edge_weight)
    out2 = layer(x, adj)
    assert torch.allclose(out1, 3.0 * x)
    assert torch.allclose(out2, 3.0 * x)
    # let's do the same using the normalized option
    # such that the output should now be
    # :math:`\mathbf{XU} + \mathbf{\hat{A}XW}`
    layer = res_gconv_factory()
    x = torch.ones(2, 1)
    edge_index = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]])
    adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(2, 2))
    edge_weight = torch.tensor([1.0, 1.0, 1.0, 1.0])
    out1 = layer(x, adj, edge_weight)
    out2 = layer(x, adj, edge_weight)
    assert torch.allclose(out1, 2.0 * x)
    assert torch.allclose(out2, 2.0 * x)


"""
def test_gcn2_conv():
    x = torch.randn(4, 16)
    x_0 = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    value = torch.rand(edge_index.size(1))
    adj1 = to_torch_csc_tensor(edge_index, size=(4, 4))
    adj2 = to_torch_csc_tensor(edge_index, value, size=(4, 4))

    conv = GCN2Conv(16, alpha=0.2)
    assert str(conv) == 'GCN2Conv(16, alpha=0.2, beta=1.0)'
    out1 = conv(x, x_0, edge_index)
    assert out1.size() == (4, 16)
    assert torch.allclose(conv(x, x_0, adj1.t()), out1, atol=1e-6)
    out2 = conv(x, x_0, edge_index, value)
    assert out2.size() == (4, 16)
    assert torch.allclose(conv(x, x_0, adj2.t()), out2, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj3 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        adj4 = SparseTensor.from_edge_index(edge_index, value, (4, 4))
        assert torch.allclose(conv(x, x_0, adj3.t()), out1, atol=1e-6)
        assert torch.allclose(conv(x, x_0, adj4.t()), out2, atol=1e-6)

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit(x, x_0, edge_index), out1, atol=1e-6)
        assert torch.allclose(jit(x, x_0, edge_index, value), out2, atol=1e-6)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit(x, x_0, adj3.t()), out1, atol=1e-6)
            assert torch.allclose(jit(x, x_0, adj4.t()), out2, atol=1e-6)

    conv.cached = True
    conv(x, x_0, edge_index)
    assert conv._cached_edge_index is not None
    assert torch.allclose(conv(x, x_0, edge_index), out1, atol=1e-6)
    assert torch.allclose(conv(x, x_0, adj1.t()), out1, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        conv(x, x_0, adj3.t())
        assert conv._cached_adj_t is not None
        assert torch.allclose(conv(x, x_0, adj3.t()), out1, atol=1e-6)
"""
