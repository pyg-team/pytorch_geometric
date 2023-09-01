import warnings
from typing import Any

import torch_geometric

if torch_geometric.typing.WITH_PT20:  # pragma: no cover
    from torch._dynamo import is_compiling
else:

    def is_compiling():
        return False


def warn(*args: Any, **kwargs: Any) -> None:
    if is_compiling():
        return

    warnings.warn(*args, **kwargs)
