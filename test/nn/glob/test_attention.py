import torch
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq

from torch_geometric.nn import GlobalAttention


def test_global_attention():
    channels, batch_size = (32, 10)
    gate_nn = Seq(Lin(channels, channels), ReLU(), Lin(channels, 1))
    nn = Seq(Lin(channels, channels), ReLU(), Lin(channels, channels))

    glob = GlobalAttention(gate_nn, nn)
    assert glob.__repr__() == (
        'GlobalAttention(gate_nn=Sequential(\n'
        '  (0): Linear(in_features=32, out_features=32, bias=True)\n'
        '  (1): ReLU()\n'
        '  (2): Linear(in_features=32, out_features=1, bias=True)\n'
        '), nn=Sequential(\n'
        '  (0): Linear(in_features=32, out_features=32, bias=True)\n'
        '  (1): ReLU()\n'
        '  (2): Linear(in_features=32, out_features=32, bias=True)\n'
        '))')

    x = torch.randn((batch_size**2, channels))
    batch = torch.arange(batch_size, dtype=torch.long)
    batch = batch.view(-1, 1).repeat(1, batch_size).view(-1)

    assert glob(x, batch).size() == (batch_size, channels)
    assert glob(x, batch, batch_size + 1).size() == (batch_size + 1, channels)
