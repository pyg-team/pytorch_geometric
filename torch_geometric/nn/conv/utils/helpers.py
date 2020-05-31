import torch


def unsqueeze(src: torch.Tensor, dim: int, length: int) -> torch.Tensor:
    for _ in range(length):
        src = src.unsqueeze(dim)
    return src
