import torch
from torch_geometric.nn import MsgNorm


def test_msg_norm():
    norm = MsgNorm(True)
    assert norm.__repr__() == ('MsgNorm(True)')
    x = torch.randn(100, 16)
    msg = torch.randn(100, 16)
    out = norm(x, msg)
    assert out.size() == (100, 16)

    norm = MsgNorm(False)
    assert norm.__repr__() == ('MsgNorm(False)')
    x = torch.randn(100, 16)
    msg = torch.randn(100, 16)
    out = norm(x, msg)
    assert out.size() == (100, 16)
