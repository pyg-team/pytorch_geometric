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


def merge_masks(
    *,
    key_pad: Optional[Tensor] = None,
    attn: Optional[Tensor] = None,
    num_heads: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    detach: bool = True,
) -> Optional[Tensor]:
    """Merge key-padding and attention masks, convert them to additive form,
    and flatten the head dimension.

    Args:
        key_pad (Optional[Tensor]): Either a 2-D tensor (B, L) as a padding
        mask or a 4-D tensor (B, H, L, L). If 2-D, num_heads must be provided.
        attn (Optional[Tensor]): Additional structural mask. Same accepted
        shapes as key_pad but 2-D is unsupported.
        num_heads (Optional[int]): Required when key_pad is a 2-D tensor.
        dtype (torch.dtype): Floating point type for the output mask.
        Defaults to torch.float32.
        detach (bool): If True, the returned mask is detached from the graph.

    Returns:
        Optional[Tensor]: Additive mask of shape (B * H, L, L). Valid positions
        are 0 and masked ones are -inf, or None if both inputs are None.
    """
    if key_pad is None and attn is None:
        return None

    if key_pad is not None and key_pad.dim() == 2:
        if num_heads is None:
            raise ValueError(
                "num_heads=... is required when key_pad has shape (B, L)")
        key_pad = build_key_padding(key_pad, num_heads=num_heads)

    merged = None
    if key_pad is not None:
        merged = _to_additive(key_pad, dtype=dtype)
    if attn is not None:
        additive = _to_additive(attn, dtype=dtype)
        merged = additive if merged is None else merged + additive

    if merged is None:  # pragma: no cover – sanity guard
        return None
    b, h, l, _ = merged.shape
    merged = merged.reshape(b * h, l, l)

    return merged.detach() if detach else merged
