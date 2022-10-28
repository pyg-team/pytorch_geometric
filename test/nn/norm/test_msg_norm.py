import torch

from torch_geometric.nn import MessageNorm
from torch_geometric.testing import is_full_test


def test_message_norm():
    norm = MessageNorm(True)
    assert norm.__repr__() == ('MessageNorm(learn_scale=True)')
    x = torch.randn(100, 16)
    msg = torch.randn(100, 16)
    assert norm(x, msg).size() == (100, 16)

    if is_full_test():
        jit = torch.jit.script(norm)
        assert jit(x, msg).size() == (100, 16)

    norm = MessageNorm(False)
    assert norm.__repr__() == ('MessageNorm(learn_scale=False)')
    x = torch.randn(100, 16)
    msg = torch.randn(100, 16)
    assert norm(x, msg).size() == (100, 16)

    if is_full_test():
        jit = torch.jit.script(norm)
        assert jit(x, msg).size() == (100, 16)
