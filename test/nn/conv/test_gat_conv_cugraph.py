# pylint: disable=too-many-arguments, too-many-locals
import pytest
import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import GATConv, GATConvCuGraph

options = {
    "add_self_loops": [True, False],
    "bias": [True, False],
    "bipartite": [True, False],
    "concat": [True, False],
    "heads": [1, 2, 3],
    "max_num_neighbors": [8, None],
    "use_sparse": [True, False],
}

device = 'cuda:0'


@pytest.mark.parametrize("use_sparse", options["use_sparse"])
@pytest.mark.parametrize("max_num_neighbors", options["max_num_neighbors"])
@pytest.mark.parametrize("heads", options["heads"])
@pytest.mark.parametrize("concat", options["concat"])
@pytest.mark.parametrize("bipartite", options["bipartite"])
@pytest.mark.parametrize("bias", options["bias"])
@pytest.mark.parametrize("add_self_loops", options["add_self_loops"])
def test_gat_conv_equality(add_self_loops, bias, bipartite, concat, heads,
                           max_num_neighbors, use_sparse):
    in_channels, out_channels = 5, 2
    args = (in_channels, out_channels, heads)
    kwargs = {"add_self_loops": add_self_loops, "bias": bias, "concat": concat}

    u = torch.tensor([7, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 8, 9])
    v = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7])
    n_id = torch.tensor([0, 1, 2, 4, 5, 6, 8, 9, 7, 3])

    if bipartite:
        num_nodes = (10, 8)
        x = torch.rand(num_nodes[0], in_channels).to(device)
    else:
        u = n_id[u]
        v = n_id[v]
        num_nodes = 10
        x = torch.rand(num_nodes, in_channels).to(device)
    edge_index = torch.stack([u, v]).to(device)

    if use_sparse:
        edge_index_sparse = SparseTensor(row=edge_index[0], col=edge_index[1])

    torch.manual_seed(12345)
    conv1 = GATConv(*args, **kwargs).to(device)
    torch.manual_seed(12345)
    conv2 = GATConvCuGraph(*args, **kwargs).to(device)

    with torch.no_grad():
        conv2.lin.weight.data[:, :] = conv1.lin_src.weight.data
        conv2.att.data[:heads * out_channels] = conv1.att_src.data.flatten()
        conv2.att.data[heads * out_channels:] = conv1.att_dst.data.flatten()

    if bipartite:
        out1 = conv1((x, x[:num_nodes[1]]), edge_index)
    else:
        out1 = conv1(x, edge_index)

    if use_sparse:
        out2 = conv2(x, edge_index_sparse, max_num_neighbors=8)
    else:
        out2 = conv2(x, edge_index, num_nodes=num_nodes,
                     max_num_neighbors=max_num_neighbors)
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
