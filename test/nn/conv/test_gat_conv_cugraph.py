import pytest
import torch

from torch_geometric.nn import GATConv, GATConvCuGraph

options = {
    "bias": [True, False],
    "concat": [True, False],
    "heads": [1, 2, 3],
}

device = 'cuda:0'


@pytest.mark.parametrize("heads", options["heads"])
@pytest.mark.parametrize("concat", options["concat"])
@pytest.mark.parametrize("bias", options["bias"])
def test_gat_conv_equality(bias, concat, heads):
    in_channels, out_channels = 5, 2
    args = (in_channels, out_channels, heads)
    kwargs = {"add_self_loops": False, "bias": bias, "concat": concat}
    u = torch.tensor([0, 1, 0, 2, 3, 0, 4, 0, 5, 0, 6, 7, 0, 8, 9])
    v = torch.tensor([1, 9, 2, 9, 9, 4, 9, 5, 9, 6, 9, 9, 8, 9, 0])
    num_nodes = 10
    edge_index = torch.stack([u, v]).to(device)
    x = torch.rand(num_nodes, in_channels).to(device)

    torch.manual_seed(12345)
    conv1 = GATConv(*args, **kwargs).to(device)
    out1 = conv1(x, edge_index)

    torch.manual_seed(12345)
    conv2 = GATConvCuGraph(*args, **kwargs).to(device)

    with torch.no_grad():
        conv2.lin.weight.data[:, :] = conv1.lin_src.weight.data
        conv2.att.data[:heads * out_channels] = conv1.att_src.data.flatten()
        conv2.att.data[heads * out_channels:] = conv1.att_dst.data.flatten()
    out2 = conv2(x, edge_index, num_nodes)
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
