# pylint: disable=too-many-arguments, too-many-locals
from collections import OrderedDict
from itertools import product

import pytest
import torch
from torch_sparse import SparseTensor

from torch_geometric.nn import RGCNConv, RGCNConvCuGraph
from torch_geometric.testing import onlyCUDA

options = OrderedDict({
    "aggr": ["add", "sum", "mean"],
    "bias": [True, False],
    "bipartite": [True, False],
    "max_num_neighbors": [8, None],
    "num_bases": [2, None],
    "root_weight": [True, False],
    "use_sparse": [True, False],
})


@onlyCUDA
@pytest.mark.parametrize(",".join(options.keys()), product(*options.values()))
def test_rgcn_conv_equality(aggr, bias, bipartite, max_num_neighbors,
                            num_bases, root_weight, use_sparse):
    device = 'cuda:0'
    in_channels, out_channels, num_relations = 4, 2, 3
    args = (in_channels, out_channels, num_relations)
    kwargs = {
        'aggr': aggr,
        'bias': bias,
        'num_bases': num_bases,
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
    edge_type = torch.tensor([1, 2, 1, 0, 2, 1, 2, 0, 2, 2, 1, 1, 1, 2,
                              2]).to(device)

    if use_sparse:
        edge_index_sparse = SparseTensor(row=edge_index[0], col=edge_index[1],
                                         value=edge_type)

    torch.manual_seed(12345)
    conv1 = RGCNConv(*args, **kwargs).to(device)
    torch.manual_seed(12345)
    conv2 = RGCNConvCuGraph(*args, **kwargs).to(device)

    if bipartite:
        out1 = conv1((x, x[:num_nodes[1]]), edge_index, edge_type)
    else:
        out1 = conv1(x, edge_index, edge_type)

    if use_sparse:
        out2 = conv2(x, edge_index_sparse, max_num_neighbors=8)
    else:
        out2 = conv2(x, edge_index, edge_type, num_nodes, max_num_neighbors)

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
