from typing import Iterable, Mapping, Optional, Tuple

import torch
from torch.nn import Module


# `torch.nn.ModuleDict` doesn't allow `.` to be used in key names.
# This `ModuleDict` will support it by converting the `.` to `#` in the
# internal representation and converts it back to `.` in the external
# representation.
class ModuleDict(torch.nn.ModuleDict):
    def __init__(self, modules: Optional[Mapping[str, Module]] = None):
        if modules is not None:  # Replace the keys in modules:
            modules = {
                self.to_internal_key(key): module
                for key, module in modules.items()
            }
        super().__init__(modules)

    @staticmethod
    def to_internal_key(key: str) -> str:
        return key.replace(".", "#")

    @staticmethod
    def to_external_key(key: str) -> str:
        return key.replace("#", ".")

    def __getitem__(self, key: str) -> Module:
        return super().__getitem__(self.to_internal_key(key))

    def __setitem__(self, key: str, module: Module):
        return super().__setitem__(self.to_internal_key(key), module)

    def __delitem__(self, key: str):
        return super().__delitem__(self.to_internal_key(key))

    def __contains__(self, key: str) -> bool:
        return super().__contains__(self.to_internal_key(key))

    def keys(self) -> Iterable[str]:
        return [self.to_external_key(key) for key in super().keys()]

    def items(self) -> Iterable[Tuple[str, Module]]:
        return [(self.to_external_key(key), value)
                for key, value in super().items()]
