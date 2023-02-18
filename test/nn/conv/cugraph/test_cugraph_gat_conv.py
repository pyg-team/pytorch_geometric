import pytest
import torch

from torch_geometric.nn import CuGraphGATConv, GATConv
from torch_geometric.testing import onlyCUDA, withPackage


@onlyCUDA
@withPackage('pylibcugraphops>=23.02')
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('bipartite', [True, False])
@pytest.mark.parametrize('concat', [True, False])
@pytest.mark.parametrize('heads', [1, 2, 3])
@pytest.mark.parametrize('max_num_neighbors', [8, None])
def test_gat_conv_equality(bias, bipartite, concat, heads, max_num_neighbors):
    in_channels, out_channels = (5, 2)
    kwargs = dict(bias=bias, concat=concat)

    size = (10, 8) if bipartite else (10, 10)
    x = torch.rand(size[0], in_channels, device='cuda')
    edge_index = torch.tensor([
        [7, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 8, 9],
        [0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7],
    ], device='cuda')

    conv1 = GATConv(in_channels, out_channels, heads, add_self_loops=False,
                    **kwargs).cuda()
    conv2 = CuGraphGATConv(in_channels, out_channels, heads, **kwargs).cuda()

    with torch.no_grad():
        conv2.lin.weight.data[:, :] = conv1.lin_src.weight.data
        conv2.att.data[:heads * out_channels] = conv1.att_src.data.flatten()
        conv2.att.data[heads * out_channels:] = conv1.att_dst.data.flatten()

    if bipartite:
        out1 = conv1((x, x[:size[1]]), edge_index)
    else:
        out1 = conv1(x, edge_index)

    csc = CuGraphGATConv.to_csc(edge_index, size)
    out2 = conv2(x, csc, max_num_neighbors=max_num_neighbors)
    assert torch.allclose(out1, out2, atol=1e-6)

    grad_output = torch.rand_like(out1)
    out1.backward(grad_output)
    out2.backward(grad_output)

    assert torch.allclose(conv1.lin_src.weight.grad, conv2.lin.weight.grad,
                          atol=1e-6)
    assert torch.allclose(conv1.att_src.grad.flatten(),
                          conv2.att.grad[:heads * out_channels], atol=1e-6)
    assert torch.allclose(conv1.att_dst.grad.flatten(),
                          conv2.att.grad[heads * out_channels:], atol=1e-6)
    if bias:
        assert torch.allclose(conv1.bias.grad, conv2.bias.grad, atol=1e-6)
