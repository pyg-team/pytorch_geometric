import pytest
import torch

from torch_geometric.nn import RGCNConv, RGCNConvCuGraph

options = {
    "aggr": ["add", "sum", "mean"],
    "bias": [True, False],
    "num_bases": [2, None],
    "root_weight": [True, False],
}

device = 'cuda:0'


@pytest.mark.parametrize('root_weight', options['root_weight'])
@pytest.mark.parametrize('num_bases', options['num_bases'])
@pytest.mark.parametrize('bias', options['bias'])
@pytest.mark.parametrize('aggr', options['aggr'])
def test_rgcn_conv_equality(aggr, bias, num_bases, root_weight):
    in_channels, out_channels, num_relations = 4, 2, 3
    args = (in_channels, out_channels, num_relations)
    kwargs = {
        'aggr': aggr,
        'bias': bias,
        'num_bases': num_bases,
        'root_weight': root_weight,
    }
    u = torch.tensor([0, 1, 0, 2, 3, 0, 4, 0, 5, 0, 6, 7, 0, 8, 9])
    v = torch.tensor([1, 9, 2, 9, 9, 4, 9, 5, 9, 6, 9, 9, 8, 9, 0])
    num_nodes = 10
    edge_index = torch.stack([u, v]).to(device)
    edge_type = torch.tensor([1, 2, 1, 0, 2, 1, 2, 0, 2, 2, 1, 1, 1, 2,
                              2]).to(device)
    x = torch.rand(num_nodes, in_channels).to(device)

    torch.manual_seed(12345)
    conv1 = RGCNConv(*args, **kwargs).to(device)
    torch.manual_seed(12345)
    conv2 = RGCNConvCuGraph(*args, **kwargs).to(device)

    out1 = conv1(x, edge_index, edge_type)
    out2 = conv2(x, edge_index, num_nodes, edge_type)

    assert torch.allclose(out1, out2, atol=1e-6)

    grad_out = torch.rand_like(out1)
    out1.backward(grad_out)
    out2.backward(grad_out)

    end = -1 if root_weight else None
    assert torch.allclose(conv1.weight.grad, conv2.weight.grad[:end],
                          atol=1e-6)

    if root_weight:
        assert torch.allclose(conv1.root.grad, conv2.weight.grad[-1],
                              atol=1e-6)

    if num_bases is not None:
        assert torch.allclose(conv1.comp.grad, conv2.comp.grad, atol=1e-6)
