from typing import Any, Tuple, Type, Union

import torch

import torch_geometric.typing

if torch_geometric.typing.WITH_PT20:
    import torch._dynamo


def is_torch_instance(obj: Any, cls: Union[Type, Tuple[Type]]) -> bool:
    r"""Checks if the :obj:`obj` is an instance of a :obj:`cls`.

    This function extends :meth:`isinstance` to be applicable during
    :meth:`torch.compile` usage by checking against the original class of
    compiled models.
    """
    # `torch.compile` removes the model inheritance and converts the model to
    # a `torch._dynamo.OptimizedModule` instance, leading to `isinstance` being
    # unable to check the model's inheritance. This function unwraps the
    # compiled model before evaluating via `isinstance`.
    if (torch_geometric.typing.WITH_PT20
            and isinstance(obj, torch._dynamo.OptimizedModule)):
        return isinstance(obj._orig_mod, cls)
    return isinstance(obj, cls)
