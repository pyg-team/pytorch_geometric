import torch
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq

from torch_geometric.nn.inits import (
    glorot,
    glorot_orthogonal,
    ones,
    reset,
    uniform,
    zeros,
)


def test_inits():
    x = torch.empty(1, 4)

    uniform(size=4, value=x)
    assert x.min() >= -0.5
    assert x.max() <= 0.5

    glorot(x)
    assert x.min() >= -1.1
    assert x.max() <= 1.1

    glorot_orthogonal(x, scale=1.0)
    assert x.min() >= -2.5
    assert x.max() <= 2.5

    zeros(x)
    assert x.tolist() == [[0, 0, 0, 0]]

    ones(x)
    assert x.tolist() == [[1, 1, 1, 1]]

    nn = Lin(16, 16)
    uniform(size=4, value=nn.weight)
    assert nn.weight[0].min() >= -0.5
    assert nn.weight[0].max() <= 0.5

    glorot(nn.weight)
    assert nn.weight[0].min() >= -0.45
    assert nn.weight[0].max() <= 0.45

    glorot_orthogonal(nn.weight, scale=1.0)
    assert nn.weight[0].min() >= -2.5
    assert nn.weight[0].max() <= 2.5


def test_reset():
    nn = Lin(16, 16)
    w = nn.weight.clone()
    reset(nn)
    assert not torch.allclose(nn.weight, w)

    nn = Seq(Lin(16, 16), ReLU(), Lin(16, 16))
    w_1, w_2 = nn[0].weight.clone(), nn[2].weight.clone()
    reset(nn)
    assert not torch.allclose(nn[0].weight, w_1)
    assert not torch.allclose(nn[2].weight, w_2)
