import copy

import torch

from torch_geometric.nn import DirGNNConv, SAGEConv
from torch_geometric.testing import is_full_test


def test_dir_gnn_conv():
    conv = DirGNNConv(SAGEConv(16, 32))
    assert str(conv) == 'DirGNNConv(SAGEConv(16, 32, aggr=mean), α=0.5)'


def test_dir_gnn_conv_with_decomposed_layers():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = DirGNNConv(SAGEConv(16, 32))

    decomposed_conv = copy.deepcopy(conv)
    decomposed_conv.decomposed_layers = 2

    out1 = conv(x, edge_index)
    out2 = decomposed_conv(x, edge_index)
    assert torch.allclose(out1, out2)

    if is_full_test():
        t = '(Tensor, Tensor, OptTensor) -> Tensor'
        jit = torch.jit.script(decomposed_conv.jittable(t))
        assert torch.allclose(jit(x, edge_index), out1)


def test_static_dir_gnn_conv():
    x = torch.randn(3, 4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = DirGNNConv(SAGEConv(16, 32))
    out = conv(x, edge_index)
    assert out.size() == (3, 4, 32)
