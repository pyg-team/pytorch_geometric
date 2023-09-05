import warnings

import torch_geometric

if torch_geometric.typing.WITH_PT20:  # pragma: no cover
    from torch._dynamo import is_compiling as _is_compiling
else:

    def _is_compiling() -> bool:  # pragma: no cover
        return False


def warn(message: str):
    if _is_compiling():
        return

    warnings.warn(message)


def filterwarnings(action: str, message: str):
    if _is_compiling():
        return

    warnings.filterwarnings(action, message)
