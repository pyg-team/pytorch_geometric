import torch

from torch_geometric.nn.aggr.utils import MultiheadAttentionBlock


def test_multihead_attention_block():
    x = torch.randn(2, 4, 8)
    y = torch.randn(2, 3, 8)
    x_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    y_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)

    block = MultiheadAttentionBlock(8, heads=2)
    block.reset_parameters()
    assert str(block) == 'MultiheadAttentionBlock(8, heads=2, layer_norm=True)'

    out = block(x, y, x_mask, y_mask)
    assert out.size() == (2, 4, 8)

    jit = torch.jit.script(block)
    assert torch.allclose(jit(x, y, x_mask, y_mask), out)
