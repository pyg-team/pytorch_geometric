from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class Connect(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, mapping: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Returns a coarsened `edge_index` and `edge_attr`"""
        pass

    @abstractmethod
    def reset_parameters(self):
        pass

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)
