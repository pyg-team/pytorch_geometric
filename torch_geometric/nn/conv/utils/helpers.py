import torch


@torch.jit.script
def abs_dim(src: torch.Tensor, dim: int) -> int:
    return src.dim() + dim if dim < 0 else dim


@torch.jit.script
def unsqueeze(src: torch.Tensor, dim: int, length: int) -> torch.Tensor:
    for _ in range(length):
        src = src.unsqueeze(dim)
    return src


@torch.jit.script
def unsqueeze_and_expand(src: torch.Tensor, dim: int,
                         size: int) -> torch.Tensor:
    src = src.unsqueeze(dim)
    sizes = [-1] * src.dim()
    sizes[dim] = size
    return src.expand(sizes)
