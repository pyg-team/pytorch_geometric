import warnings

import torch_geometric

if torch_geometric.typing.WITH_PT20:  # pragma: no cover
    from torch._dynamo import is_compiling
else:

    def is_compiling() -> bool:
        return False


def warn(*args, **kwargs):
    if is_compiling():
        return

    warnings.warn(*args, **kwargs)


def filterwarnings(*args, **kwargs):
    if is_compiling():
        return

    warnings.filterwarnings(*args, **kwargs)
