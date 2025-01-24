from typing import Final, Iterable, Mapping, Optional, Tuple, Union

import torch
from torch.nn import Module

Key = Union[str, Tuple[str, ...]]


# `torch.nn.ModuleDict` doesn't allow `.` to be used in key names.
# This `ModuleDict` will support it by converting the `.` to `#` in the
# internal representation and converts it back to `.` in the external
# representation. It also allows passing tuples as keys.
class ModuleDict(torch.nn.ModuleDict):
    CLASS_ATTRS: Final[Tuple[str, ...]] = tuple(dir(torch.nn.ModuleDict))

    def __init__(
        self,
        modules: Optional[Mapping[Union[str, Tuple[str, ...]], Module]] = None,
    ):
        if modules is not None:  # Replace the keys in modules:
            modules = {
                self.to_internal_key(key): module
                for key, module in modules.items()
            }
        super().__init__(modules)

    @classmethod
    def to_internal_key(cls, key: Key) -> str:
        if isinstance(key, tuple):  # ModuleDict can't handle tuples as keys
            assert len(key) > 1
            key = f"<{'___'.join(key)}>"
        assert isinstance(key, str)

        # ModuleDict cannot handle keys that exists as class attributes:
        if key in cls.CLASS_ATTRS:
            key = f'<{key}>'

        # ModuleDict cannot handle dots in keys:
        return key.replace('.', '#')

    @classmethod
    def to_external_key(cls, key: str) -> Key:
        key = key.replace('#', '.')

        if key[0] == '<' and key[-1] == '>' and key[1:-1] in cls.CLASS_ATTRS:
            key = key[1:-1]

        if key[0] == '<' and key[-1] == '>' and '___' in key:
            key = tuple(key[1:-1].split('___'))

        return key

    def __getitem__(self, key: Key) -> Module:
        return super().__getitem__(self.to_internal_key(key))

    def __setitem__(self, key: Key, module: Module):
        return super().__setitem__(self.to_internal_key(key), module)

    def __delitem__(self, key: Key):
        return super().__delitem__(self.to_internal_key(key))

    def __contains__(self, key: Key) -> bool:
        return super().__contains__(self.to_internal_key(key))

    def keys(self) -> Iterable[Key]:
        return [self.to_external_key(key) for key in super().keys()]

    def items(self) -> Iterable[Tuple[Key, Module]]:
        return [(self.to_external_key(k), v) for k, v in super().items()]
