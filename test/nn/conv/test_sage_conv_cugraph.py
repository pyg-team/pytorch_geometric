import pytest
import torch

from torch_geometric.nn import SAGEConv, SAGEConvCuGraph

options = {
    "aggr": ["sum", "mean", "min", "max"],
    "bias": [True, False],
    "normalize": [True, False],
    "root_weight": [True, False],
}

device = 'cuda:0'


@pytest.mark.parametrize("root_weight", options["root_weight"])
@pytest.mark.parametrize("normalize", options["normalize"])
@pytest.mark.parametrize("bias", options["bias"])
@pytest.mark.parametrize("aggr", options["aggr"])
def test_sage_conv_equality(aggr, bias, normalize, root_weight):
    in_channels, out_channels = 4, 2
    args = (in_channels, out_channels)
    kwargs = {
        'aggr': aggr,
        'bias': bias,
        'normalize': normalize,
        'root_weight': root_weight,
    }
    u = torch.tensor([0, 1, 0, 2, 3, 0, 4, 0, 5, 0, 6, 7, 0, 8, 9])
    v = torch.tensor([1, 9, 2, 9, 9, 4, 9, 5, 9, 6, 9, 9, 8, 9, 0])
    num_nodes = 10
    edge_index = torch.stack([u, v]).to(device)
    x = torch.rand(num_nodes, in_channels).to(device)

    torch.manual_seed(12345)
    conv1 = SAGEConv(*args, **kwargs).to(device)
    torch.manual_seed(12345)
    conv2 = SAGEConvCuGraph(*args, **kwargs).to(device)

    with torch.no_grad():
        conv2.linear.weight.data[:, :in_channels] = conv1.lin_l.weight.data
        if root_weight:
            conv2.linear.weight.data[:, in_channels:] = conv1.lin_r.weight.data
        if bias:
            conv2.linear.bias.data[:] = conv1.lin_l.bias.data

    out1 = conv1(x, edge_index)
    out2 = conv2(x, edge_index, num_nodes)
    assert torch.allclose(out1, out2, atol=1e-6)

    grad_out = torch.rand_like(out1)
    out1.backward(grad_out)
    out2.backward(grad_out)
    assert torch.allclose(conv1.lin_l.weight.grad,
                          conv2.linear.weight.grad[:, :in_channels], atol=1e-6)

    if root_weight:
        assert torch.allclose(conv1.lin_r.weight.grad,
                              conv2.linear.weight.grad[:, in_channels:],
                              atol=1e-6)
    if bias:
        assert torch.allclose(conv1.lin_l.bias.grad, conv2.linear.bias.grad,
                              atol=1e-6)
