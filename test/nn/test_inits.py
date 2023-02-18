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
    assert x.min() >= -1.25
    assert x.max() <= 1.25

    glorot_orthogonal(x, scale=1.0)
    assert x.min() >= -1.25
    assert x.max() <= 1.25

    zeros(x)
    assert x.tolist() == [[0, 0, 0, 0]]

    ones(x)
    assert x.tolist() == [[1, 1, 1, 1]]

    nn = Lin(16, 16)
    uniform(size=4, value=nn.weight)
    assert min(nn.weight.tolist()[0]) >= -0.5
    assert max(nn.weight.tolist()[0]) <= 0.5

    glorot(nn.weight)
    assert min(nn.weight.tolist()[0]) >= -1.25
    assert max(nn.weight.tolist()[0]) <= 1.25

    glorot_orthogonal(nn.weight, scale=1.0)
    assert min(nn.weight.tolist()[0]) >= -1.25
    assert max(nn.weight.tolist()[0]) <= 1.25


def test_reset():
    nn = Lin(16, 16)
    w = nn.weight.clone()
    reset(nn)
    assert not nn.weight.tolist() == w.tolist()

    nn = Seq(Lin(16, 16), ReLU(), Lin(16, 16))
    w_1, w_2 = nn[0].weight.clone(), nn[2].weight.clone()
    reset(nn)
    assert not nn[0].weight.tolist() == w_1.tolist()
    assert not nn[2].weight.tolist() == w_2.tolist()
