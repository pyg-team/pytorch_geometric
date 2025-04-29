from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor

from torch_geometric.data import Data


class BasePositionalEncoder(nn.Module, ABC):
    """Base class for all positional encoders."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, data: Data) -> Tensor:
        """Compute positional encodings for the given graph data.

        Args:
            data (Data): The input graph data object.

        Returns:
            Tensor: Positional encodings aligned with data.x
        """
        raise NotImplementedError
