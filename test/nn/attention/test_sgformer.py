import torch

from torch_geometric.nn.attention import SGFormerAttention


def test_sgformer_attention():
    x = torch.randn(1, 4, 16)
    attn = SGFormerAttention(channels=16, heads=1)
    out = attn(x)
    assert out.shape == (4, 64)
    assert str(attn) == ('SGFormerAttention(heads=1, '
                         'head_channels=64)')
