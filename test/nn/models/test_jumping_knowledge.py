import torch

from torch_geometric.nn import JumpingKnowledge
from torch_geometric.testing import is_full_test


def test_jumping_knowledge():
    num_nodes, channels, num_layers = 100, 17, 5
    xs = list([torch.randn(num_nodes, channels) for _ in range(num_layers)])

    model = JumpingKnowledge('cat')
    assert model.__repr__() == 'JumpingKnowledge(cat)'

    out = model(xs)
    assert out.size() == (num_nodes, channels * num_layers)

    if is_full_test():
        jit = torch.jit.script(model)
        assert torch.allclose(jit(xs), out)

    model = JumpingKnowledge('max')
    assert model.__repr__() == 'JumpingKnowledge(max)'

    out = model(xs)
    assert out.size() == (num_nodes, channels)

    if is_full_test():
        jit = torch.jit.script(model)
        assert torch.allclose(jit(xs), out)

    model = JumpingKnowledge('lstm', channels, num_layers)
    assert model.__repr__() == 'JumpingKnowledge(lstm)'

    out = model(xs)
    assert out.size() == (num_nodes, channels)

    if is_full_test():
        jit = torch.jit.script(model)
        assert torch.allclose(jit(xs), out)
