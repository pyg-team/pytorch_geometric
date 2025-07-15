import pytest  # Ensure pytest is imported if not already
import torch

from torch_geometric.contrib.utils.mask_utils import (
    build_key_padding,
    merge_masks,
)


def test_build_key_padding_two_graphs(two_graph_batch):
    """Test key padding mask generation for two-graph batch.

    Verifies that the padding mask has correct shape and expected values when
    processing a batch containing two graphs.
    """
    batch_vec = two_graph_batch.batch
    mask = build_key_padding(batch_vec, num_heads=2)
    assert mask.shape == (2, 2, 3, 3)
    assert not mask[0, 0, 0, 1]
    assert not mask[0, 0, 0, 2]
    assert not mask[1, 0, 1, 2]


def test_build_key_padding_single_graph(simple_batch):
    """Test key padding mask generation for single-graph batch.

    Verifies that when processing a batch with a single graph, the resulting
    padding mask contains no padding values.
    """
    batch = simple_batch(8, 3).batch
    mask = build_key_padding(batch, num_heads=1)

    assert mask.shape == (1, 1, 3, 3)
    assert not mask.any()


def test_build_key_padding_empty():
    """Test key padding mask generation for empty batch.

    Verifies that processing an empty batch tensor results in a zero-sized mask
    with appropriate dimensions.
    """
    batch = torch.tensor([], dtype=torch.long)
    mask = build_key_padding(batch, num_heads=1)
    assert mask.shape == (0, 1, 0, 0)


def test_merge_masks_both_masks(two_graph_batch):
    """Test merge_masks with both key padding and attention masks.

    Verifies correct merging behavior when both key padding and attention
    boolean masks are provided as inputs.
    """
    batch_vec = two_graph_batch.batch
    key_pad = build_key_padding(batch_vec, num_heads=2)
    attn = torch.zeros((2, 2, 3, 3), dtype=torch.bool)
    attn[0, 0, 0, 1] = True

    merged = merge_masks(key_pad=key_pad, attn=attn)
    assert merged.shape == (4, 3, 3)
    assert torch.isinf(merged[0, 0, 1])


def test_merge_masks_only_key_pad(two_graph_batch):
    """Test merge_masks functionality with only key padding mask.

    Verifies that the function correctly expands and returns the key padding
    mask when no attention mask is provided.
    """
    batch_vec = two_graph_batch.batch
    key_pad = build_key_padding(batch_vec, num_heads=1)

    merged = merge_masks(key_pad=key_pad, attn=None)
    assert merged.shape == (2, 3, 3)


def test_merge_masks_only_attn():
    """Test merge_masks functionality with only attention mask.

    Verifies that providing only an attention mask results in correct
    flattening of batch and head dimensions in the output.
    """
    attn = torch.zeros((2, 2, 4, 4), dtype=torch.bool)
    merged = merge_masks(key_pad=None, attn=attn)
    assert merged.shape == (4, 4, 4)


def test_merge_masks_none_none():
    """Test merge_masks behavior when no masks are provided.

    Verifies that the function returns None when both key_pad and attn masks
    are set to None.
    """
    assert merge_masks(key_pad=None, attn=None) is None


def test_merge_masks_float_bias(simple_batch):
    """Test merge_masks with boolean key padding and float bias values.

    Verifies correct handling of combined boolean key padding mask and float
    attention bias values, ensuring proper shape and expected output values.
    """
    b1 = simple_batch(1, 2).batch
    b2 = simple_batch(1, 2).batch + 1
    batch_vec = torch.cat([b1, b2], dim=0)
    key_pad = build_key_padding(batch_vec, num_heads=1)

    bias = torch.zeros_like(key_pad, dtype=torch.float)
    bias[0, 0, 0, 1] = 5.0

    merged = merge_masks(key_pad=key_pad, attn=bias)
    assert merged.shape == (2, 2, 2)
    assert merged[0, 0, 1] == 5.0
    assert merged[1, 1, 0] == 0.0


def test_merge_masks_key_pad_missing_num_heads():
    """Test that merge_masks raises ValueError when key_pad is 2-D and
    num_heads is None.
    """
    key_pad = torch.zeros((2, 3), dtype=torch.bool)
    with pytest.raises(ValueError):
        merge_masks(key_pad=key_pad, attn=None)


def test_merge_masks_invalid_mask_dtype():
    """Test that merge_masks raises TypeError when mask dtype is invalid."""
    attn = torch.zeros((2, 2, 3, 3), dtype=torch.int)
    with pytest.raises(TypeError):
        merge_masks(key_pad=None, attn=attn)
