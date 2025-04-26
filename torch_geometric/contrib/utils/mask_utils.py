from typing import Optional

import torch
from torch import Tensor


def build_key_padding(batch: Tensor, *, num_heads: int) -> Tensor:
    """Create a boolean key-padding mask.

    Parameters
    ----------
    batch : LongTensor, shape (N,)
        Graph index per node. Nodes that share the same value belong to the
        same graph. The values **do not** need to start at 0 or be contiguous.
    num_heads : int
        Number of attention heads. The mask is broadcast along this dimension.

    Returns:
    -------
    BoolTensor
        Shape (B, num_heads, L_max, L_max) where B = #graphs and L_max is the
        size of the largest graph in the mini-batch. `True` marks *invalid*
        (masked) positions.
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
    """Convert a boolean or float mask to additive form (0 / -inf).

    Boolean `True` → `-inf`  •  Boolean `False` → `0`
    Float values are returned unchanged (assumed already additive).
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
    """Merge key-padding and attention masks, converting the result to additive
    form and flattening the head dimension.

    Parameters
    ----------
    key_pad : Tensor | None
        • BoolTensor (B, L) – per-node padding mask.  \
        • Bool / float (B, H, L, L) – already square.  \
        If 2-D, `num_heads` must be supplied.
    attn : Tensor | None
        Additional structural mask, same accepted shapes / dtypes as
        `key_pad` *except* the 2-D variant is **not** supported.
    num_heads : int | None
        Required only when `key_pad` is 2-D.
    dtype : torch.dtype
        Floating dtype for the additive mask (defaults to fp32).
    detach : bool
        If true, returned mask is detached to avoid autograd overhead.

    Returns:
    -------
    FloatTensor | None
        Additive mask of shape (B × H, L, L), or `None` if both inputs are
        `None`. Valid positions contain `0`, masked positions contain `-inf`.
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

    # (B, H, L, L) → (B*H, L, L)
    b, h, l, _ = merged.shape
    merged = merged.reshape(b * h, l, l)

    return merged.detach() if detach else merged
