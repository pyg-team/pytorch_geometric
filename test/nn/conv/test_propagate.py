import torch

from torch_geometric.nn import SAGEConv


def test_propagate():
    SAGEConv.jit_on_init = True
    conv = SAGEConv(10, 16)
    SAGEConv.jit_on_init = False

    old_propagate = conv.__class__.propagate
    old_collect = conv.__class__._collect
    conv.__class__.propagate = conv.propagate_jit
    conv.__class__._collect = conv._collect_jit
    torch.jit.script(conv)
    conv.__class__.propagate = old_propagate
    conv.__class__._collect = old_collect
