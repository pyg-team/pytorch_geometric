from typing import Any, List, Union

import torch


def normalize_string(s: str) -> str:
    return s.lower().replace('-', '').replace('_', '').replace(' ', '')


def resolver(classes: List[Any], query: Union[Any, str], *args, **kwargs):
    if query is None or not isinstance(query, str):
        return query

    query = normalize_string(query)
    for cls in classes:
        if query == normalize_string(cls.__name__):
            return cls(*args, **kwargs)

    return ValueError(
        f"Could not resolve '{query}' among the choices "
        f"{set(normalize_string(cls.__name__) for cls in classes)}")


def activation_resolver(query: Union[Any, str] = 'relu', *args, **kwargs):
    acts = [
        act for act in vars(torch.nn.modules.activation).values()
        if isinstance(act, type) and issubclass(act, torch.nn.Module)
    ]
    return resolver(acts, query, *args, **kwargs)
