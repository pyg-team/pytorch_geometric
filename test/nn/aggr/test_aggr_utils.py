import torch

from torch_geometric.nn.aggr.utils import (
    InducedSetAttentionBlock,
    MultiheadAttentionBlock,
    PoolingByMultiheadAttention,
    SetAttentionBlock,
)
from torch_geometric.testing import withCUDA


@withCUDA
def test_multihead_attention_block(device: torch.device):
    x = torch.randn(2, 4, 8, device=device)
    y = torch.randn(2, 3, 8, device=device)
    x_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool,
                          device=device)
    y_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool,
                          device=device)

    block = MultiheadAttentionBlock(8, heads=2, device=device)
    block.reset_parameters()
    assert str(block) == ('MultiheadAttentionBlock(8, heads=2, '
                          'layer_norm=True, dropout=0.0)')

    out = block(x, y, x_mask, y_mask)
    assert out.size() == (2, 4, 8)

    jit = torch.jit.script(block)
    assert torch.allclose(jit(x, y, x_mask, y_mask), out)

    # Scripted and eager must also agree for self-attention in eval mode, the
    # only case in which the MHA fastpath can be taken:
    block.eval()
    with torch.no_grad():
        out = block(x, x, x_mask, x_mask)
        assert torch.allclose(jit.eval()(x, x, x_mask, x_mask), out)


@withCUDA
def test_multihead_attention_block_mha_fastpath(device: torch.device,
                                                monkeypatch):
    # `merge_masks` is only called on the native MHA fastpath, so record
    # whether a padding mask is present whenever the fastpath is taken:
    calls = []
    merge_masks = torch.nn.MultiheadAttention.merge_masks

    def spy(self, attn_mask, key_padding_mask, query):
        calls.append(key_padding_mask is not None)
        return merge_masks(self, attn_mask, key_padding_mask, query)

    monkeypatch.setattr(torch.nn.MultiheadAttention, 'merge_masks', spy)

    x = torch.randn(2, 4, 8, device=device)
    y_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool,
                          device=device)

    block = MultiheadAttentionBlock(8, heads=2, device=device)
    block.eval()

    # Positive control: a boolean key padding mask takes the fastpath, so the
    # assertions below cannot pass vacuously with the fastpath unavailable:
    with torch.no_grad():
        block.attn(x, x, x, ~y_mask, need_weights=False)
    assert calls == [True]
    calls.clear()

    with torch.no_grad():
        out = block(x, x, y_mask=y_mask)
    assert out.size() == (2, 4, 8)
    assert out.isfinite().all()

    # Masked self-attention must avoid the fastpath on CUDA:
    if device.type == 'cuda':
        assert calls == []
    else:
        assert calls == [True]


@withCUDA
def test_multihead_attention_block_dropout(device: torch.device):
    x = torch.randn(2, 4, 8, device=device)

    block = MultiheadAttentionBlock(8, dropout=0.5, device=device)
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
