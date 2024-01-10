import torch

from torch_geometric.nn import SAGEConv


def test_propagate():
    conv = SAGEConv(10, 16)
    conv.__class__.propagate = conv.propagate_jit
    conv.__class__._collect = conv._collect_jit
    torch.jit.script(conv)
