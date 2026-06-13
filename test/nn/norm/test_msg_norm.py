import torch

from torch_geometric.nn import MessageNorm
from torch_geometric.testing import withDevice


@withDevice
def test_message_norm(device):
    norm = MessageNorm(learn_scale=True, device=device)
    assert str(norm) == 'MessageNorm(learn_scale=True)'
    x = torch.randn(100, 16, device=device)
    msg = torch.randn(100, 16, device=device)
    out = norm(x, msg)
    assert out.size() == (100, 16)

    norm = MessageNorm(learn_scale=False, device=device)
    assert str(norm) == 'MessageNorm(learn_scale=False)'
    out = norm(x, msg)
    assert out.size() == (100, 16)
