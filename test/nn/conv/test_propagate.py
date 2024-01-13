import pytest
import torch

from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_torch_csc_tensor


@pytest.mark.parametrize('fuse', [True, False])
def test_propagate(fuse):
    SAGEConv.jit_on_init = True
    conv = SAGEConv(16, 32)
    conv.fuse = fuse
    SAGEConv.jit_on_init = False

    old_propagate = conv.__class__.propagate
    old_collect = conv.__class__._collect

    try:
        conv.__class__.propagate = conv.propagate_jit
        conv.__class__._collect = conv.collect_jit
        jit = torch.jit.script(conv)

        x = torch.randn(4, 16)
        edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
        adj = to_torch_csc_tensor(edge_index, size=(4, 4))

        out1 = jit(x, edge_index)
        out2 = jit(x, adj.t())
        assert torch.allclose(out1, out2)

    finally:
        conv.__class__.propagate = old_propagate
        conv.__class__._collect = old_collect
