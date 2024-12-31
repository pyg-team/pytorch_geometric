import torch

from torch_geometric.nn.attention import PolynormerAttention


def test_polynormer_attention():
    x = torch.randn(1, 4, 16)
    attn = PolynormerAttention(channels=16, heads=4)
    out = attn(x)
    import pdb
    pdb.set_trace()
    assert out.shape == (1, 4, 16)
    assert str(attn) == ('PolynormerAttention(heads=4, '
                         'head_channels=64 kernel=ReLU())')
