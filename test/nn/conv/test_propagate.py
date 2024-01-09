import torch

from torch_geometric.nn import SAGEConv


def test_propagate():
    conv = SAGEConv(10, 16)
    jit = torch.jit.script(conv)
    print(jit)
