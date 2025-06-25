import warnings
from typing import Literal

import torch_geometric


def warn(message: str, stacklevel: int = 5) -> None:
    if torch_geometric.is_compiling():
        return

    warnings.warn(message, stacklevel=stacklevel)


def filterwarnings(
    action: Literal['default', 'error', 'ignore', 'always', 'module', 'once'],
    message: str,
) -> None:
    if torch_geometric.is_compiling():
        return

    warnings.filterwarnings(action, message)


class WarningCache(set):
    """Cache for warnings."""
    def warn(self, message: str, stacklevel: int = 5) -> None:
        """Trigger warning message."""
        if message not in self:
            self.add(message)
            warn(message, stacklevel=stacklevel)
