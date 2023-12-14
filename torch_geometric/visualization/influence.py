from typing import Any

import torch
from torch import Tensor
from torch.autograd import grad


def influence(model: torch.nn.Module, src: Tensor, *args: Any) -> Tensor:
    x = src.clone().requires_grad_()
    out = model(x, *args).sum(dim=-1)

    influences = []
    for j in range(src.size(0)):
        influence = grad([out[j]], [x], retain_graph=True)[0].abs().sum(dim=-1)
        influences.append(influence / influence.sum())

    return torch.stack(influences, dim=0)
