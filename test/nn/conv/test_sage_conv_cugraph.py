# pylint: disable=too-many-arguments, too-many-locals
import pytest
import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import SAGEConv, SAGEConvCuGraph

options = {
    "aggr": ["sum", "mean", "min", "max"],
    "bias": [True, False],
    "bipartite": [True, False],
    "max_num_neighbors": [8, None],
    "normalize": [True, False],
    "root_weight": [True, False],
    "use_sparse": [True, False],
}

device = 'cuda:0'


@pytest.mark.parametrize("use_sparse", options["use_sparse"])
@pytest.mark.parametrize("root_weight", options["root_weight"])
@pytest.mark.parametrize("normalize", options["normalize"])
@pytest.mark.parametrize("max_num_neighbors", options["max_num_neighbors"])
@pytest.mark.parametrize("bipartite", options["bipartite"])
@pytest.mark.parametrize("bias", options["bias"])
@pytest.mark.parametrize("aggr", options["aggr"])
def test_sage_conv_equality(aggr, bias, bipartite, max_num_neighbors,
                            normalize, root_weight, use_sparse):
    in_channels, out_channels = 4, 2
    args = (in_channels, out_channels)
    kwargs = {
        'aggr': aggr,
        'bias': bias,
        'normalize': normalize,
        'root_weight': root_weight,
    }

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
    conv1 = SAGEConv(*args, **kwargs).to(device)
    torch.manual_seed(12345)
    conv2 = SAGEConvCuGraph(*args, **kwargs).to(device)

    with torch.no_grad():
        conv2.linear.weight.data[:, :in_channels] = conv1.lin_l.weight.data
        if root_weight:
            conv2.linear.weight.data[:, in_channels:] = conv1.lin_r.weight.data
        if bias:
            conv2.linear.bias.data[:] = conv1.lin_l.bias.data

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
