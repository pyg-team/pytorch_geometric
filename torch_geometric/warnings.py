import warnings
from typing import Any, Literal

import torch_geometric


def warn(message: str, **kwargs: Any) -> None:
    if torch_geometric.is_compiling():
        return

    warnings.warn(message, **kwargs)


def filterwarnings(
    action: Literal['default', 'error', 'ignore', 'always', 'module', 'once'],
    message: str,
) -> None:
    if torch_geometric.is_compiling():
        return

    warnings.filterwarnings(action, message)


class WarningCache(set):
    """Cache for warnings."""
    def warn(self, message: str, stacklevel: int = 5, **kwargs: Any) -> None:
        """Trigger warning message."""
        if message not in self:
            self.add(message)
            warn(message, stacklevel=stacklevel, **kwargs)
