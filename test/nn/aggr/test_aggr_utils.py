import torch

from torch_geometric.nn.aggr.utils import (
    InducedSetAttentionBlock,
    MultiheadAttentionBlock,
    PoolingByMultiheadAttention,
    SetAttentionBlock,
)


def test_multihead_attention_block():
    x = torch.randn(2, 4, 8)
    y = torch.randn(2, 3, 8)
    x_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    y_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)

    block = MultiheadAttentionBlock(8, heads=2)
    block.reset_parameters()
    assert str(block) == ('MultiheadAttentionBlock(8, heads=2, '
                          'layer_norm=True, dropout=0.0)')

    out = block(x, y, x_mask, y_mask)
    assert out.size() == (2, 4, 8)

    jit = torch.jit.script(block)
    assert torch.allclose(jit(x, y, x_mask, y_mask), out)


def test_multihead_attention_block_dropout():
    x = torch.randn(2, 4, 8)

    block = MultiheadAttentionBlock(8, dropout=0.5)
    assert not torch.allclose(block(x, x), block(x, x))


def test_set_attention_block():
    x = torch.randn(2, 4, 8)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)

    block = SetAttentionBlock(8, heads=2)
    block.reset_parameters()
    assert str(block) == ('SetAttentionBlock(8, heads=2, layer_norm=True, '
                          'dropout=0.0)')

    out = block(x, mask)
    assert out.size() == (2, 4, 8)

    jit = torch.jit.script(block)
    assert torch.allclose(jit(x, mask), out)


def test_induced_set_attention_block():
    x = torch.randn(2, 4, 8)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)

    block = InducedSetAttentionBlock(8, num_induced_points=2, heads=2)
    assert str(block) == ('InducedSetAttentionBlock(8, num_induced_points=2, '
                          'heads=2, layer_norm=True, dropout=0.0)')

    out = block(x, mask)
    assert out.size() == (2, 4, 8)

    jit = torch.jit.script(block)
    assert torch.allclose(jit(x, mask), out)


def test_pooling_by_multihead_attention():
    x = torch.randn(2, 4, 8)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)

    block = PoolingByMultiheadAttention(8, num_seed_points=2, heads=2)
    assert str(block) == ('PoolingByMultiheadAttention(8, num_seed_points=2, '
                          'heads=2, layer_norm=True, dropout=0.0)')

    out = block(x, mask)
    assert out.size() == (2, 2, 8)

    jit = torch.jit.script(block)
    assert torch.allclose(jit(x, mask), out)
