from abc import ABC, abstractmethod
from typing import Optional

import torch


class Select(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None,
                **kwargs) -> torch.tensor:
        """Groups nodes into super nodes (clusters) and returns a
        assignment tensor of shape :obj:`[num_nodes, 2]`. Some nodes
        might not be assigned to any super node and will be assigned
        to -1."""
        pass

    @abstractmethod
    def reset_parameters(self):
        pass

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)
