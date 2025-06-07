import torch

from torch_geometric.nn.attention import PerformerAttention


def test_performer_attention():
    x = torch.randn(1, 4, 16)
    mask = torch.ones([1, 4], dtype=torch.bool)
    attn = PerformerAttention(channels=16, heads=4)
    out = attn(x, mask)
    assert out.shape == (1, 4, 16)
    assert str(attn) == ('PerformerAttention(heads=4, '
                         'head_channels=64 kernel=ReLU())')
