import torch

from torch_geometric.contrib.utils.mask_utils import (
    build_key_padding,
    merge_masks,
)


def test_build_key_padding():
    """Test the build_key_padding function to ensure it creates the correct
    key padding mask for a given batch of graphs.
    """
    # Test case 1: Simple batch with 2 graphs
    batch = torch.tensor([0, 0, 1, 1, 1])
    num_heads = 2
    mask = build_key_padding(batch, num_heads=num_heads)

    assert mask.shape == (2, 2, 3, 3)  # (B=2, H=2, L=3, L=3)
    assert not mask[0, 0, 0, 1]  # No padding within first graph
    assert mask[0, 0, 0, 2]  # Padding between graphs
    assert not mask[1, 0, 1, 2]  # No padding within second graph

    # Test case 2: Single graph
    batch = torch.tensor([0, 0, 0])
    mask = build_key_padding(batch, num_heads=1)
    assert mask.shape == (1, 1, 3, 3)
    assert not mask.any()  # No padding needed

    # Test case 3: Empty graph
    batch = torch.tensor([])
    mask = build_key_padding(batch, num_heads=1)
    assert mask.shape == (0, 1, 0, 0)


def test_merge_masks():
    """Test the merge_masks function to ensure it correctly combines
    key padding and attention masks.
    """
    # Test case 1: Both masks provided
    batch = torch.tensor([0, 0, 1, 1])
    key_pad = build_key_padding(batch, num_heads=2)
    attn = torch.zeros((2, 2, 2, 2), dtype=torch.bool)
    attn[0, 0, 0, 1] = True  # Add constraint in attention

    merged = merge_masks(key_pad=key_pad, attn=attn)
    assert merged.shape == (4, 2, 2)  # (B*H, L, L)
    assert torch.isinf(merged[0, 0, 1])  # Check constraint applied

    # Test case 2: Only key padding mask
    key_pad_mask = build_key_padding(batch, num_heads=1)
    merged = merge_masks(key_pad=key_pad_mask, attn=None)
    assert merged.shape == (2, 2, 2)  # (B*H, L, L)

    # Test case 3: Only attention mask
    attn = torch.zeros((2, 2, 3, 3), dtype=torch.bool)
    merged = merge_masks(key_pad=None, attn=attn)
    assert merged.shape == (4, 3, 3)  # (B*H, L, L)

    # Test case 4: No masks
    merged = merge_masks(key_pad=None, attn=None)
    assert merged is None


def test_merge_masks_with_bias_float():
    """Test merge_masks combining boolean key padding & float bias values."""
    batch = torch.tensor([0, 0, 1])
    key_pad = build_key_padding(batch, num_heads=1)

    # Create float bias matching key_pad shape
    bias = torch.zeros_like(key_pad, dtype=torch.float)
    bias[0, 0, 0, 1] = 5.0

    merged = merge_masks(key_pad=key_pad, attn=bias)
    assert merged.shape == (2, 2, 2)
    assert merged[0, 0, 1] == 5.0
    assert torch.isinf(merged[1, 1, 0])
