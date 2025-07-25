import pytest
import torch
from torch_geometric.data import Batch

from torch_geometric.contrib.utils.mask_utils import (
    build_key_padding,
    merge_masks,
)


def _mixed_batch(simple_batch):
    """Build a two‑graph batch of sizes [2, 1] via existing fixture."""
    return Batch.from_data_list([
        simple_batch(1, 2).to_data_list()[0],
        simple_batch(1, 1).to_data_list()[0],
    ]).batch


def test_merge_masks_boolean_padding_to_additive(simple_batch):
    """Test merging boolean key padding into an additive mask."""
    # Mixed‑length batch: graph‑0 has 2 nodes, graph‑1 has 1 node → padding
    batch_vec = Batch.from_data_list([
        simple_batch(1, 2).to_data_list()[0],
        simple_batch(1, 1).to_data_list()[0],
    ]).batch

    key_pad = build_key_padding(batch_vec, num_heads=1)  # (B=2,H=1,L=2)

    # Zero structural bias so only padding influences the result
    attn = torch.zeros_like(key_pad, dtype=torch.float32)
    merged = merge_masks(key_pad=key_pad, attn=attn, dtype=torch.float32)

    # Shape is flattened to (B*H, L, L)
    assert merged.shape == (2, 2, 2)
    # The padded row/col of graph‑1 should contain −inf
    assert torch.isinf(merged[1]).any()
    # Un‑padded entries remain finite (zero here)
    assert merged[0, 0, 1] == pytest.approx(0.0)


def test_merge_masks_float_bias_cast_to_dtype(simple_batch):
    """Test merging float structural bias with key padding,
    ensuring dtype casting.
    """
    batch_vec = Batch.from_data_list([
        simple_batch(1, 2).to_data_list()[0],
        simple_batch(1, 1).to_data_list()[0],
    ]).batch

    key_pad = build_key_padding(batch_vec, num_heads=1)

    # Structural bias supplied in float64; constant −7.5 everywhere
    attn = torch.full(key_pad.shape, -7.5, dtype=torch.float64)
    merged = merge_masks(key_pad=key_pad, attn=attn, dtype=torch.float32)

    assert merged.dtype == torch.float32
    # An un‑padded coordinate equals the original bias value after cast
    assert merged[0, 0, 0] == pytest.approx(-7.5, rel=1e-6)


def test_merge_masks_detach_flag_controls_grad(two_graph_batch):
    """`detach` flag controls whether gradients flow through `attn`."""
    batch_vec = two_graph_batch.batch
    key_pad = build_key_padding(batch_vec, num_heads=1)

    attn = torch.zeros_like(key_pad, dtype=torch.float32, requires_grad=True)
    # 1) detach=False → gradients flow
    merged_keep = merge_masks(key_pad=key_pad, attn=attn, detach=False)
    merged_keep.sum().backward()
    assert attn.grad is not None and torch.all(attn.grad != 0)

    # 2) default detach=True → no gradients
    attn.grad.zero_()
    merged_detached = merge_masks(key_pad=key_pad, attn=attn)  # default
    assert not merged_detached.requires_grad
    assert attn.grad.abs().sum() == 0


def test_merge_masks_key_pad_rank_validation():
    """Test that `merge_masks` raises ValueError for invalid key_pad ranks."""
    bad_key_pad = torch.zeros((2, 3, 3), dtype=torch.bool)  # rank‑3
    with pytest.raises(ValueError):
        merge_masks(key_pad=bad_key_pad, attn=None)


def test_merge_masks_four_d_key_pad_passthrough():
    """Supplying a 4‑D key_pad must succeed without reshaping errors."""
    key_pad4d = torch.zeros((1, 2, 3, 3), dtype=torch.bool)  # (B=1,H=2,L=3)
    merged = merge_masks(key_pad=key_pad4d, attn=None)

    # Shape flattens heads: 1×2 → 2
    assert merged.shape == (2, 3, 3)
    # All zeros because input mask was all‑False
    assert merged.abs().sum() == 0


def test_build_key_padding_empty():
    """Empty ``batch`` tensor returns an all‑empty 4‑D boolean mask."""
    empty = torch.empty(0, dtype=torch.long)
    mask = build_key_padding(empty, num_heads=2)

    assert mask.shape == (0, 2, 0, 0)
    assert mask.dtype == torch.bool


def test_merge_masks_return_none_when_no_masks():
    """Test that `merge_masks` returns None when no masks are provided."""
    assert merge_masks(key_pad=None, attn=None) is None


def test_merge_masks_key_pad_missing_num_heads_raises():
    """Test that `merge_masks` raises ValueError if key_pad is 2-D."""
    key_pad_2d = torch.zeros((2, 3), dtype=torch.bool)
    with pytest.raises(ValueError):
        merge_masks(key_pad=key_pad_2d, attn=None)


def test_merge_masks_only_key_pad(simple_batch):
    """Test merging only key_pad into an additive mask."""
    batch_vec = simple_batch(4, 3).batch  # single graph, 3 nodes
    key_pad = build_key_padding(batch_vec, num_heads=3)  # 3 heads

    merged = merge_masks(key_pad=key_pad, attn=None, dtype=torch.float32)
    assert merged.shape == (3, 3, 3)
    assert torch.isfinite(merged).all()


def test_merge_masks_only_attn():
    """Test merging only ``attn`` into an additive mask."""
    batch_size = 2
    num_heads = 2
    seq_length = 3
    attn_bool = torch.zeros((batch_size, num_heads, seq_length, seq_length),
                            dtype=torch.bool)
    attn_bool[:, :, 0, 1] = True

    merged = merge_masks(key_pad=None, attn=attn_bool, dtype=torch.float32)

    assert merged.shape == (batch_size * num_heads, seq_length, seq_length)
    assert torch.isinf(merged[0, 0, 1])


def test_merge_masks_invalid_mask_dtype_raises():
    """Test that `merge_masks` raises TypeError for non-float `attn`."""
    attn_int = torch.zeros((1, 1, 2, 2), dtype=torch.int32)
    with pytest.raises(TypeError):
        merge_masks(attn=attn_int)
