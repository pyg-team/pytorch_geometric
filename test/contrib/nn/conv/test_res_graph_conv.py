import pytest
import torch

import torch_geometric.typing
from torch_geometric.contrib.nn.conv.res_graph_conv import (
    ResGConv,
    res_gconv_norm,
)
from torch_geometric.nn.conv import GCN2Conv
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    sort_edge_index,
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
    edge_weight = torch.tensor([1.0, 0.0, -1.0, 1.0])
    assert not is_torch_sparse_tensor(edge_index)
    expected_edge_index = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]])
    expected_edge_weight = 0.5 * torch.tensor([1.0, 0.0, -1.0, 1.0])
    expected_edge_index, expected_edge_weight = sort_edge_index(
        expected_edge_index, expected_edge_weight)
    edge_index_out, edge_weight_out = res_gconv_norm(edge_index, edge_weight,
                                                     add_self_loops=False)
    edge_index_out, edge_weight_out = sort_edge_index(edge_index_out,
                                                      edge_weight_out)
    assert torch.equal(edge_index_out, expected_edge_index)
    assert torch.allclose(expected_edge_weight, edge_weight_out)

    # this is for the case where
    # edge_index passed to res_gconv_norm
    # isinstance(edge_index, SparseTensor)
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, edge_weight,
                                           sparse_sizes=(2, 2))
        expected_adj = 0.5 * torch.tensor([[1.0, -1.0], [0.0, 1.0]])
        adj_out = res_gconv_norm(adj, add_self_loops=False)
        assert torch.equal(adj_out.to_dense(), expected_adj)
        # check that addding the edge weights via res_gconv_
        # norm doesn't have the same effect
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(2, 2))
        expected_adj = 0.5 * torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        adj_out = res_gconv_norm(adj, edge_weight=edge_weight,
                                 add_self_loops=False)
        assert torch.equal(adj_out.to_dense(), expected_adj)

    # this is for the case where
    # edge_index pass to res_gconv_norm
    # is_torch_sparse_tensor(edge_index)
    adj = to_torch_csr_tensor(edge_index, edge_weight)
    expected_edge_index = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]])
    expected_edge_weight = 0.5 * torch.tensor([1.0, 0.0, -1.0, 1.0])
    adj_out, none = res_gconv_norm(adj, add_self_loops=False)
    expected_edge_index, expected_edge_weight = sort_edge_index(
        expected_edge_index, expected_edge_weight)
    edge_index_out, edge_weight_out = to_edge_index(adj_out)
    edge_index_out, edge_weight_out = sort_edge_index(edge_index_out,
                                                      edge_weight_out)
    assert torch.equal(edge_index_out, expected_edge_index)
    assert torch.equal(edge_weight_out, expected_edge_weight)

    # check that addding out of bound edge weights
    # don't influence the results, i.e. they are
    # not taken into account if the adjacency matrix
    # is provided via a sparse is_torch_sparse_tensor.
    adj = to_torch_csr_tensor(edge_index)
    expected_edge_index = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]])
    expected_edge_weight = 0.5 * torch.tensor([1.0, 1.0, 1.0, 1.0])
    adj_out, none = res_gconv_norm(
        adj, edge_weight=torch.tensor([4.0, 4.0, 3.0, 4.0]),
        add_self_loops=False)
    expected_edge_index, expected_edge_weight = sort_edge_index(
        expected_edge_index, expected_edge_weight)
    edge_index_out, edge_weight_out = to_edge_index(adj_out)
    edge_index_out, edge_weight_out = sort_edge_index(edge_index_out,
                                                      edge_weight_out)
    assert torch.equal(edge_index_out, expected_edge_index)
    assert torch.equal(edge_weight_out, expected_edge_weight)


@pytest.fixture
def res_gconv_factory():
    def _create_layer(channels: int = 1, linear_fill: float = 1.0,
                      bias_fill: float = 1.0, bias_u: bool = False,
                      bias_w: bool = False, add_self_loops: bool = False,
                      cached: bool = False, normalize: bool = True, **kwargs):
        """Fixture providing a res_gconv layer with fixed
        weights for testing.
        """
        layer = ResGConv(channels=channels, bias_u=bias_u, bias_w=bias_w,
                         add_self_loops=add_self_loops, cached=cached,
                         normalize=normalize, **kwargs)
        with torch.no_grad():
            layer.U.weight.fill_(linear_fill)
            if layer.U.bias is not None:
                layer.U.bias.fill_(bias_fill)
            layer.W.weight.fill_(linear_fill)
            if layer.W.bias is not None:
                layer.W.bias.fill_(bias_fill)
        return layer

    return _create_layer


@pytest.mark.parametrize(
    "normalize,bias_u,bias_w,expected_output_ww_1,"
    "expected_output_ww_2,expected_output_wow_1,expected_output_wow_2", [
        (True, False, False, 1.5, 1.0, 2.0, 2.0),
        (True, True, True, 3.0, 2.0, 4.0, 4.0),
        (True, False, True, 2.0, 1.0, 3.0, 3.0),
        (True, True, False, 2.5, 2.0, 3.0, 3.0),
        (False, False, False, 2.0, 1.0, 3.0, 3.0),
        (False, True, True, 4.0, 2.0, 6.0, 6.0),
        (False, False, True, 3.0, 1.0, 5.0, 5.0),
        (False, True, False, 3.0, 2.0, 4.0, 4.0),
    ])
def test_res_gconv(res_gconv_factory, normalize, bias_u, bias_w,
                   expected_output_ww_1, expected_output_ww_2,
                   expected_output_wow_1, expected_output_wow_2):
    # we create our simple layer for the test
    # it contains U, W = 1.0, 1.0 (both a single channel, i.e. scalar)
    # as well as biases b_u = 1.0, b_w = 1.0 if set to true
    layer = res_gconv_factory(normalize=normalize, bias_u=bias_u,
                              bias_w=bias_w)

    # test the branch where
    # isinstance(edge_index, Tensor)
    #
    # we setup a dummy input where the first index
    # denotes the number of nodes in the graph (2)
    # and the second one denotes the hidden dimension
    # the res_gconv layer expects (1) from the fixture
    # above. We test the general expression
    # :math:`\mathbf{XU+b_u} + \mathbf{\hat{A}(XW+b_w)}`.
    #
    x = torch.ones(2, 1)
    edge_index = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]])
    edge_weight = torch.tensor([1.0, 0.0, -1.0, 1.0])
    out1 = layer(x, edge_index, edge_weight)
    out2 = layer(x, edge_index)
    # test against the output calculated by hand
    expected_output_ww = torch.tensor([[expected_output_ww_1],
                                       [expected_output_ww_2]])
    expected_output_wow = torch.tensor([[expected_output_wow_1],
                                        [expected_output_wow_2]])
    assert torch.allclose(out1, expected_output_ww)
    assert torch.allclose(out2, expected_output_wow)

    # test the branch where
    # isinstance(edge_index, SparseTensor)
    #
    # we setup a dummy input where the first index
    # denotes the number of nodes in the graph (2)
    # and the second one denotes the hidden dimension
    # the res_gconv layer expects (1) from the fixture
    # above. We test the general expression
    # :math:`\mathbf{XU+b_u} + \mathbf{\hat{A}(XW+b_w)}`.
    # Here, adding the edge_weight tensor should have
    # no effect, since we only focus on the SparseTensor
    # passed to forward.
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        layer = res_gconv_factory(normalize=normalize, bias_u=bias_u,
                                  bias_w=bias_w)
        x = torch.ones(2, 1)
        adj_dense = torch.tensor([[1.0, -1.0], [0.0, 1.0]])
        adj = SparseTensor.from_dense(adj_dense)
        out1 = layer(x, adj.t(), edge_weight)
        out2 = layer(x, adj.t())
        assert torch.allclose(out1, expected_output_ww)
        # check for no effect of the SparseTensor
        assert torch.allclose(out2, expected_output_ww)

    # the following tests that the caching is working correctly
    layer.cached = True
    layer(x, edge_index)
    if normalize:
        assert layer._cached_edge_index is not None
    else:
        assert layer._cached_edge_index is None

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        layer(x, adj)
        if normalize:
            assert layer._cached_adj_t is not None
        else:
            assert layer._cached_edge_index is None

    # to check that we didn't make a mistake with a transpose or so, we check
    # our layer agains the GCN2Conv
    if bias_u is False and bias_w is False and normalize is False:
        adj_gcn2conv_dense = torch.tensor([[2.0, -2.0], [0.0, 2.0]])
        adj_gcn2conv = SparseTensor.from_dense(adj_gcn2conv_dense)
        gcn2conv_layer = GCN2Conv(1, alpha=0.5, theta=0.0, layer=1,
                                  normalize=normalize, add_self_loops=False)
        ours = layer(x, adj.t())
        gcn2conv_ref = gcn2conv_layer(x, 2 * x, adj_gcn2conv.t())
        assert torch.allclose(ours, gcn2conv_ref)

    # finally, a quick test for the string
    assert str(layer) == f'ResGConv({1}, bias_u={bias_u}, bias_w={bias_w})'


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
