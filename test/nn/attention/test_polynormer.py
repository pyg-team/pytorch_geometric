import pytest
import torch

from torch_geometric.nn.attention import PolynormerAttention


@pytest.mark.parametrize('qk_shared', [True, False])
def test_performer_attention(qk_shared):
    x = torch.randn(1, 4, 16)
    mask = torch.ones([1, 4], dtype=torch.bool)
    attn = PolynormerAttention(channels=16, heads=4, qk_shared=qk_shared)
    out = attn(x, mask)
    assert out.shape == (1, 4, 256)
    assert str(attn) == 'PolynormerAttention(heads=4, head_channels=64)'
