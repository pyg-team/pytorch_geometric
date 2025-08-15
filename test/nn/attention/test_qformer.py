import torch

from torch_geometric.nn.attention import QFormer


def test_qformer():
    x = torch.randn(1, 4, 16)
    attn = QFormer(input_dim=16, hidden_dim=16, output_dim=32, num_heads=4,
                   num_layers=2)
    out = attn(x)

    assert out.shape == (1, 4, 32)
    assert str(attn) == ('QFormer(num_heads=4, num_layers=2)')
