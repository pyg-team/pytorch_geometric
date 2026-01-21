from typing import Optional

import torch
from torch import Tensor


def build_key_padding(batch: Tensor, *, num_heads: int) -> Tensor:
    """Generate a boolean mask indicating padded positions for each graph.

    Args:
        batch (LongTensor): Graph index per node. Nodes with the same value
            belong to the same graph. Values need not start at 0 or be
            contiguous.
        num_heads (int): Number of attention heads. The mask is broadcast along
            the head dimension.

    Returns:
        BoolTensor: Tensor of shape (B, num_heads, L_max, L_max) where B is the
            number of graphs and L_max is the size of the largest graph. True
            indicates masked positions.
    """
    if batch.numel() == 0:
        return torch.empty((0, num_heads, 0, 0), dtype=torch.bool,
                           device=batch.device)
    num_graphs = int(batch.max()) + 1
    lengths = torch.bincount(batch, minlength=num_graphs)
    l_max = int(lengths.max())
    positions = torch.arange(l_max, device=batch.device)
    base_mask = positions.unsqueeze(0) >= lengths.unsqueeze(1)
    square_mask = base_mask.unsqueeze(1) | base_mask.unsqueeze(2)
    return square_mask.unsqueeze(1).expand(-1, num_heads, -1, -1).clone()


def _to_additive(mask: Tensor, *, dtype: torch.dtype) -> Tensor:
    """Convert a boolean or float mask to additive form (0 or -inf).

    Args:
        mask (Tensor): Boolean mask (True for masked positions) or float mask.
            If float, the mask is assumed to be in additive form.
        dtype (torch.dtype): Floating point data type to use.

    Returns:
        Tensor: Additive mask where True is converted to -inf and False to 0.
            Float masks are cast to the specified dtype.
    """
    if mask.dtype == torch.bool:
        out = torch.zeros_like(mask, dtype=dtype)
        out.masked_fill_(mask, float("-inf"))
        return out
    if torch.is_floating_point(mask):
        return mask.to(dtype)
    raise TypeError("Mask tensor must be boolean or floating point, "
                    f"got dtype {mask.dtype}")


def _expand_key_pad(key_pad: Tensor, *, num_heads: int) -> Tensor:
    """Ensure key_pad is 4-D, expanding 2-D masks along heads.

    Args:
        key_pad (Tensor): 2-D (B, L) or 4-D (B, H, L, L) tensor.
        num_heads (int): Number of attention heads.

    Returns:
        Tensor: 4-D key padding mask.
    """
    if key_pad.dim() == 4:
        return key_pad
    if key_pad.dim() == 2:
        return build_key_padding(key_pad, num_heads=num_heads)
    raise ValueError("key_pad must be rank 2 or 4")


def _combine_additive(
    key_add: Optional[Tensor],
    attn_add: Optional[Tensor],
) -> Optional[Tensor]:
    """Sum two additive masks, handling ``None`` inputs.

    Args:
        key_add (Optional[Tensor]): Additive key-padding mask.
        attn_add (Optional[Tensor]): Additive structural mask.

    Returns:
        Optional[Tensor]: Combined additive mask or ``None``.
    """
    if key_add is None:
        return attn_add
    if attn_add is None:
        return key_add
    return key_add + attn_add


def merge_masks(
    *,
    key_pad: Optional[Tensor] = None,
    attn: Optional[Tensor] = None,
    num_heads: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    detach: bool = True,
) -> Optional[Tensor]:
    """Merge key-padding and structural masks into an additive form.

    Args:
        key_pad (Optional[Tensor]): 2-D (B, L) or 4-D (B, H, L, L) mask.
        attn (Optional[Tensor]): 4-D (B, H, L, L) mask.
        num_heads (Optional[int]): Required if ``key_pad`` is 2-D.
        dtype (torch.dtype): Output dtype. Defaults to ``torch.float32``.
        detach (bool): If True, the result is detached.

    Returns:
        Optional[Tensor]: Additive mask of shape (B*H, L, L) or ``None``.
    """
    if key_pad is None and attn is None:
        return None

    key_add = None
    if key_pad is not None:
        if key_pad.dim() == 2 and num_heads is None:
            raise ValueError("num_heads must be provided for 2-D key_pad")
        key_pad = _expand_key_pad(key_pad, num_heads=num_heads or 1)
        key_add = _to_additive(key_pad, dtype=dtype)

    attn_add = _to_additive(attn, dtype=dtype) if attn is not None else None
    merged = _combine_additive(key_add, attn_add)

    if merged is None:
        return None
    b, h, l, _ = merged.shape
    merged = merged.reshape(b * h, l, l)
    return merged.detach() if detach else merged
