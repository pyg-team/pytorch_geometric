import torch

from torch_geometric.nn.attention import PolynormerAttention


def test_polynormer_attention():
    x = torch.randn(4, 16)
    attn = PolynormerAttention(channels=16, heads=4)
    out = attn(x)
    assert out.shape == (4, 256)
    assert str(attn) == ('PolynormerAttention(heads=4, '
                         'head_channels=64)')
