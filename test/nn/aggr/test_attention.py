import torch
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq

from torch_geometric.nn.aggr import AttentionAggregation
from torch_geometric.nn.glob import GlobalAttention


def test_attention_aggregation():
    channels, dim_size = (32, 10)
    gate_nn = Seq(Lin(channels, channels), ReLU(), Lin(channels, 1))
    nn = Seq(Lin(channels, channels), ReLU(), Lin(channels, channels))

    aggr = AttentionAggregation(gate_nn, nn)
    assert aggr.__repr__() == (
        'AttentionAggregation(gate_nn=Sequential(\n'
        '  (0): Linear(in_features=32, out_features=32, bias=True)\n'
        '  (1): ReLU()\n'
        '  (2): Linear(in_features=32, out_features=1, bias=True)\n'
        '), nn=Sequential(\n'
        '  (0): Linear(in_features=32, out_features=32, bias=True)\n'
        '  (1): ReLU()\n'
        '  (2): Linear(in_features=32, out_features=32, bias=True)\n'
        '))')

    x = torch.randn((dim_size**2, channels))
    index = torch.arange(dim_size, dtype=torch.long)
    index = index.view(-1, 1).repeat(1, dim_size).view(-1)

    assert aggr(x, index).size() == (dim_size, channels)
    assert aggr(x, index,
                dim_size=dim_size + 1).size() == (dim_size + 1, channels)

    # test depreciated
    aggr = GlobalAttention(gate_nn, nn)
    assert aggr(x, index).size() == (dim_size, channels)
    assert aggr(x, index, dim_size + 1).size() == (dim_size + 1, channels)
