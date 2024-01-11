import torch

from torch_geometric.nn import SAGEConv


def test_propagate():
    conv = SAGEConv(10, 16)

    old_propagate = conv.__class__.propagate
    old_collect = conv.__class__._collect
    conv.__class__.propagate = conv.propagate_jit
    conv.__class__._collect = conv._collect_jit
    torch.jit.script(conv)
    conv.__class__.propagate = old_propagate
    conv.__class__._collect = old_collect
